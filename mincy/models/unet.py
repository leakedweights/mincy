import jax
import jax.numpy as jnp
from flax import linen as nn

from typing import Optional, Callable
from functools import partial

from ..components.blocks import ResnetBlock, AttentionBlock, Upsample, Downsample, FourierEmbedding, sinusoidal_emb


class UNet(nn.Module):
    channel_mults: tuple
    attention_mults: tuple
    kernel_size: tuple
    dropout: float
    num_init_channels: int
    num_res_blocks: int
    pos_emb_type: str
    pos_emb_dim: int
    rescale_skip_conns: bool
    class_cond: bool
    num_classes: int
    resblock_variant: str
    nonlinearity: Callable
    downsample_last_dim: bool = False
    sandwich_attention: bool = True
    fourier_scale: Optional[float] = None

    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array, t: jax.Array, train: bool = True):
        b, h, w, c = x.shape

        ParameterizedResBlock = partial(ResnetBlock,
                                        variant=self.resblock_variant,
                                        kernel_size=self.kernel_size,
                                        rescale=self.rescale_skip_conns,
                                        nonlinearity=self.nonlinearity,
                                        dropout=self.dropout)

        if self.class_cond:
            class_emb = nn.Embed(self.num_classes, self.pos_emb_dim)

        input_layer = nn.Conv(features=self.num_init_channels,
                              kernel_size=self.kernel_size, padding="SAME")

        down_layers = list()
        up_layers = list()

        for i, channel_mult in enumerate(self.channel_mults):
            is_last = i == len(self.channel_mults) - 1
            layer = []
            num_channels = self.num_init_channels * channel_mult

            for _ in range(self.num_res_blocks):
                layer.append(ParameterizedResBlock(
                    features=num_channels
                ))

                if channel_mult in self.attention_mults and self.sandwich_attention:
                    layer.append(
                        AttentionBlock()
                    )

            if channel_mult in self.attention_mults and not self.sandwich_attention:
                layer.append(AttentionBlock())

            if self.downsample_last_dim or not is_last:
                layer.append(ParameterizedResBlock(
                    features=num_channels,
                    transform=Downsample(
                        out_channels=num_channels, kernel_size=self.kernel_size),
                    conv_transform=Downsample(
                        out_channels=num_channels, kernel_size=self.kernel_size)
                ))

            down_layers.append(layer)

        num_bottom_channels = self.num_init_channels * self.channel_mults[-1]

        bottom_layers = [
            ParameterizedResBlock(features=num_bottom_channels),
            AttentionBlock(),
            ParameterizedResBlock(features=num_bottom_channels)]

        for i, channel_mult in enumerate(list(reversed(self.channel_mults))):
            layer = []
            num_channels = self.num_init_channels * channel_mult

            for _ in range(self.num_res_blocks + 1):
                layer.append(ParameterizedResBlock(
                    features=num_channels
                ))

                if channel_mult in self.attention_mults and not self.sandwich_attention:
                    layer.append(AttentionBlock())

            if self.downsample_last_dim or i != 0:
                layer.append(ParameterizedResBlock(
                    features=num_channels,
                    transform=Upsample(
                        out_channels=num_channels, kernel_size=self.kernel_size),
                    conv_transform=Upsample(
                        out_channels=num_channels, kernel_size=self.kernel_size)
                ))

            up_layers.append(layer)

        output_layer = nn.Conv(
            features=c, kernel_size=self.kernel_size, padding="SAME")

        if self.pos_emb_type == "fourier":
            assert self.fourier_scale is not None, "Fourier scale must be specified when using Gaussian Fourier embeddings!"
            pos_emb = FourierEmbedding(self.pos_emb_dim, self.fourier_scale)(t)
        elif self.pos_emb_type == "sinusoidal":
            pos_emb = sinusoidal_emb(t, self.pos_emb_dim)
        else:
            raise NotImplementedError(
                f"Embedding type '{self.pos_emb_type}' not supported.")

        if self.class_cond:
            pos_emb = pos_emb + class_emb(y)

        residuals = []

        x = input_layer(x)

        for layer in down_layers:
            for layer_block in layer:
                x = layer_block(x, pos_emb, deterministic=not train)
            residuals.append(x)

        for layer in bottom_layers:
            x = layer(x, pos_emb, deterministic=not train)

        for layer in up_layers:
            x = jnp.concatenate([x, residuals.pop()], axis=-1)
            for layer_block in layer:
                x = layer_block(x, pos_emb, deterministic=not train)

        x = output_layer(x)

        return x
