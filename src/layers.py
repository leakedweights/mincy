import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Sequence, Optional, Callable, Any

Shape = int | Sequence[int]


class Upsample(nn.Module):
    kernel_size: tuple
    out_channels: int = None

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
    out_channels: int = None

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        out_channels = self.out_channels if self.out_channels is not None else c

        x = nn.Conv(features=out_channels,
                    kernel_size=self.kernel_size,
                    strides=(2, 2))(x)
        return x


class ConvBlock(nn.Module):
    dim: int
    kernel_size: Shape
    dropout: float
    num_groups: int
    transform: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic: bool):
        if not isinstance(self.num_groups, int):
            jax.debug.print("type: {t}", t=type(self.num_groups))

        x = nn.GroupNorm(self.num_groups)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic)
        x = nn.silu(x)

        if self.transform is not None:
            x = self.transform(x)

        x = nn.Conv(self.dim, kernel_size=self.kernel_size)(x)

        return x


class ResnetBlock(nn.Module):
    dim: int
    kernel_size: Shape
    num_groups: int
    dropout: float
    time_embed_dim: int
    transform: Optional[Callable] = None
    conv_transform: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, t: int, deterministic: bool):
        time_embed = nn.Dense(features=self.time_embed_dim)(t)
        time_embed = nn.silu(time_embed)
        time_embed = nn.Dense(features=self.dim)(time_embed)
        time_embed = time_embed[:, None, None, :]

        h = ConvBlock(dim=self.dim,
                      kernel_size=self.kernel_size,
                      num_groups=self.num_groups,
                      dropout=self.dropout,
                      transform=self.conv_transform)(x, deterministic)

        h += time_embed

        if self.transform is not None:
            x = self.transform(x)
        elif x.shape != h.shape:
            x = nn.Conv(features=self.dim, kernel_size=self.kernel_size)(x)

        return x + h


class AttentionBlock(nn.Module):
    channels: int
    num_heads: int
    head_dim: int
    dropout: float

    @nn.compact
    def __call__(self, x, t: jnp.ndarray, deterministic: bool):
        b, h_size, w, c = x.shape
        x = nn.GroupNorm()(x)
        x_flat = x.reshape((b, -1, c))

        qkv = nn.Dense(features=(self.num_heads * 3 * self.head_dim),
                       kernel_init=jax.nn.initializers.normal(0.02), name='qkv')(x_flat)
        qkv = qkv.reshape((b, -1, self.num_heads, 3 * self.head_dim))
        query, key, value = jnp.split(qkv, 3, axis=-1)

        query = query / jnp.sqrt(self.head_dim)
        query = query.transpose((0, 2, 1, 3))
        key = key.transpose((0, 2, 1, 3))
        value = value.transpose((0, 2, 1, 3))

        attn_weights = jax.nn.softmax(jnp.einsum(
            'bhqd, bhkd -> bhqk', query, key), axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout)(
            attn_weights, deterministic=deterministic)
        attn_output = jnp.einsum('bhqk, bhvd -> bhqd', attn_weights, value)
        attn_output = attn_output.transpose((0, 2, 1, 3)).reshape(
            (b, -1, self.num_heads * self.head_dim))

        h = nn.Dense(features=self.channels, kernel_init=jax.nn.initializers.normal(
            0.02), name='out')(attn_output)
        h = h.reshape((b, h_size, w, self.channels))
        return x + h
