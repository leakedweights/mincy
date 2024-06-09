import jax
import jax.numpy as jnp
import flax.linen as nn

import string
from typing import Optional, Callable, Sequence, Any

Shape = int | Sequence


class Upsample(nn.Module):
    kernel_size: tuple
    out_channels: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape

        out_channels = self.out_channels if self.out_channels is not None else c

        x = jax.image.resize(x, (b, 2 * h, 2 * w, c), "nearest")
        x = nn.Conv(features=out_channels,
                    kernel_size=self.kernel_size)(x)
        return x


class Downsample(nn.Module):
    kernel_size: tuple
    out_channels: Optional[int] = None

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        out_channels = self.out_channels if self.out_channels is not None else c

        x = nn.Conv(features=out_channels,
                    kernel_size=self.kernel_size,
                    strides=(2, 2))(x)
        return x


class ConvBlock(nn.Module):
    features: int
    kernel_size: Shape
    dropout: float
    nonlinearity: Callable
    transform: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic: bool):

        num_groups = min(x.shape[-1] // 4, 32)
        x = nn.GroupNorm(num_groups)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = self.nonlinearity(x)

        if self.transform is not None:
            x = self.transform(x)

        x = nn.Conv(self.features, kernel_size=self.kernel_size)(x)

        return x


class ResnetBlock(nn.Module):
    variant: str
    rescale: bool
    features: int
    kernel_size: Any
    dropout: float
    nonlinearity: Callable
    transform: Optional[Callable] = None
    conv_transform: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, pos_emb, deterministic: bool = False):
        variants = ["BigGAN++"]

        if self.variant not in variants:
            raise NotImplementedError(
                f"Unrecognized variant. Currently supported: {', '.join(variants)}.")

        if self.variant == "BigGAN++":
            pos_emb = self.nonlinearity(pos_emb)
            pos_embed_proj = nn.DenseGeneral(self.features)(pos_emb)[
                :, None, None, :]

            h = x

            if self.transform is not None or x.shape[-1] != self.features:
                if self.transform is not None:
                    x = self.transform(x)
                x = nn.Conv(self.features, [1, 1])(x)

            h = ConvBlock(self.features, self.kernel_size, dropout=0.0,
                          nonlinearity=self.nonlinearity, transform=self.conv_transform)(h, deterministic)
            h += pos_embed_proj
            h = ConvBlock(features=self.features, kernel_size=self.kernel_size,
                          nonlinearity=self.nonlinearity, dropout=self.dropout)(h, deterministic)

            if self.rescale:
                x = (x + h) / jnp.sqrt(2)
            else:
                x = x + h

            return x


def default_init(scale=1.0):
    scale = 1e-10 if scale == 0 else scale
    return jax.nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


def _einsum(a, b, c, x, y):
    einsum_str = "{},{}->{}".format("".join(a), "".join(b), "".join(c))
    return jnp.einsum(einsum_str, x, y)


def contract_inner(x, y):
    x_chars = list(string.ascii_lowercase[: len(x.shape)])
    y_chars = list(string.ascii_uppercase[: len(y.shape)])
    assert len(x_chars) == len(x.shape) and len(y_chars) == len(y.shape)
    y_chars[0] = x_chars[-1]
    out_chars = x_chars[:-1] + y_chars[1:]
    return _einsum(x_chars, y_chars, out_chars, x, y)


class NIN(nn.Module):
    """https://arxiv.org/abs/2011.13456"""

    num_units: int
    init_scale: float = 0.1

    @nn.compact
    def __call__(self, x):
        in_dim = int(x.shape[-1])
        W = self.param(
            "W", default_init(scale=self.init_scale), (in_dim, self.num_units)
        )
        b = self.param("b", jax.nn.initializers.zeros, (self.num_units,))
        y = contract_inner(x, W) + b
        assert y.shape == x.shape[:-1] + (self.num_units,)
        return y


class AttentionBlock(nn.Module):
    init_scale: float = 0.0
    variant: str = "NCSN++"

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        if self.variant == "NCSN++":
            def transform(units, init_scale=0.1): return NIN(units, init_scale)
        else:
            transform = nn.Identity()

        b, h, w, c = x.shape
        num_groups = min(c // 4, 32)

        attn = nn.GroupNorm(num_groups)(x)
        Q = transform(c)(x)
        K = transform(c)(x)
        V = transform(c)(x)

        attn = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / (c * jnp.sqrt(1/2))
        attn = nn.softmax(attn / jnp.sqrt(c), axis=-1)
        attn = jnp.matmul(attn, V)
        attn = transform(c, init_scale=self.init_scale)(attn)

        return attn + x
