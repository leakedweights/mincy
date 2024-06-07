import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Optional, Callable, Sequence, Any

Shape = int | Sequence


class ConvBlock(nn.Module):
    features: int
    kernel_size: Shape
    dropout: float
    num_groups: int
    nonlinearity: Callable
    transform: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic: bool):
        x = nn.GroupNorm(self.num_groups)(x)
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
    num_groups: int
    pos_embed_dim: int
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
            pos_embed_proj = nn.DenseGeneral(self.pos_embed_dim)(pos_emb)[
                :, None, None, :]
            
            h = x

            if self.transform is not None:
                x = nn.Conv(self.features, [1, 1])(x)
                x = self.transform(x)

            h = ConvBlock(self.features, self.kernel_size, dropout=0, num_groups=self.num_groups,
                          nonlinearity=self.nonlinearity, transform=self.conv_transform)(h, deterministic)
            h += pos_embed_proj
            h = ConvBlock(features=self.features, kernel_size=self.kernel_size, num_groups=self.num_groups,
                          nonlinearity=self.nonlinearity, dropout=self.dropout)(h, deterministic)

            if self.rescale:
                x = (x + h) / jnp.sqrt(2)
            else:
                x = x + h

            return x