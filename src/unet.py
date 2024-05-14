import jax
from jax import random
import jax.numpy as jnp
import flax.linen as nn

from layers import *

from dataclasses import dataclass
from typing import Sequence, Optional, Callable, Any

Shape = int | Sequence[int]


def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = jnp.log(10000) / (half_dim - 1)
    emb = jnp.exp(jnp.arange(half_dim) * -emb)
    emb = timesteps[:, None] * emb[None, :]
    emb = jnp.concatenate([jnp.sin(emb), jnp.cos(emb)], axis=-1)
    if embedding_dim % 2 == 1:
        emb = jnp.pad(emb, [(0, 0), (0, 1)])
    return emb


@dataclass
class UNetConfig:
    input_dim: int | Sequence
    dim_mults: tuple
    attention_mults: tuple
    time_embed_dim: int
    num_init_channels: int
    num_resnet_blocks: int
    num_norm_groups: int
    num_attention_heads: int
    attention_head_dim: int
    downsample_last: bool
    dropout: float
    kernel_size: tuple


class UNet(nn.Module):
    config: UNetConfig

    @nn.compact
    def __call__(self, x: jax.Array, t: jax.Array, train: bool):

        dim_mults = self.config.dim_mults
        attention_mults = self.config.attention_mults
        num_resnet_blocks = self.config.num_resnet_blocks
        num_norm_groups = self.config.num_norm_groups
        num_heads = self.config.num_attention_heads
        head_dim = self.config.attention_head_dim
        time_embed_dim = self.config.time_embed_dim
        num_init_channels = self.config.num_init_channels
        downsample_last = self.config.downsample_last
        dropout = self.config.dropout
        kernel_size = self.config.kernel_size

        input_channels = self.config.input_dim[-1]

        input_layer = nn.Conv(
            num_init_channels, kernel_size=kernel_size)

        down_layers = list()
        up_layers = list()

        for i, dim_mult in enumerate(dim_mults):
            is_last = i == len(dim_mults) - 1
            layer = []
            num_channels = num_init_channels * dim_mult

            for _ in range(num_resnet_blocks):
                layer.append(ResnetBlock(
                    dim=num_channels,
                    kernel_size=kernel_size,
                    num_groups=num_norm_groups,
                    dropout=dropout,
                    time_embed_dim=time_embed_dim
                ))

                if dim_mult in attention_mults:
                    layer.append(
                        AttentionBlock(channels=num_channels,
                                       num_heads=num_heads,
                                       head_dim=head_dim,
                                       dropout=dropout)
                    )

            if downsample_last or not is_last:
                layer.append(ResnetBlock(
                    dim=num_channels,
                    kernel_size=kernel_size,
                    num_groups=num_norm_groups,
                    dropout=dropout,
                    time_embed_dim=time_embed_dim,
                    transform=Downsample(
                        out_channels=num_channels, kernel_size=kernel_size),
                    conv_transform=Downsample(
                        out_channels=num_channels, kernel_size=kernel_size)
                ))

            down_layers.append(layer)

        num_bottom_channels = num_init_channels * dim_mults[-1]

        bottom_layers = [
            ResnetBlock(
                dim=num_bottom_channels,
                kernel_size=kernel_size,
                num_groups=num_norm_groups,
                dropout=dropout,
                time_embed_dim=time_embed_dim
            ),
            AttentionBlock(channels=num_bottom_channels,
                           num_heads=num_heads,
                           head_dim=head_dim,
                           dropout=dropout),
            ResnetBlock(
                dim=num_bottom_channels,
                kernel_size=kernel_size,
                num_groups=num_norm_groups,
                dropout=dropout,
                time_embed_dim=time_embed_dim
            )]

        for i, dim_mult in enumerate(list(reversed(dim_mults))):
            layer = []
            num_channels = num_init_channels * dim_mult

            for _ in range(num_resnet_blocks + 1):
                layer.append(ResnetBlock(
                    dim=num_channels,
                    kernel_size=kernel_size,
                    num_groups=num_norm_groups,
                    dropout=dropout,
                    time_embed_dim=time_embed_dim
                ))

                if dim_mult in attention_mults:
                    layer.append(
                        AttentionBlock(channels=num_channels,
                                       num_heads=num_heads,
                                       head_dim=head_dim,
                                       dropout=dropout)
                    )

            if downsample_last or i != 0:
                layer.append(ResnetBlock(
                    dim=num_channels,
                    kernel_size=kernel_size,
                    num_groups=num_norm_groups,
                    dropout=dropout,
                    time_embed_dim=time_embed_dim,
                    transform=Upsample(
                        out_channels=num_channels, kernel_size=kernel_size),
                    conv_transform=Upsample(
                        out_channels=num_channels, kernel_size=kernel_size)
                ))

            up_layers.append(layer)

        output_layer = ConvBlock(dim=input_channels,
                                 kernel_size=kernel_size,
                                 dropout=dropout,
                                 num_groups=num_norm_groups,
                                 transform=None)

        # forward pass
        t_emb = get_timestep_embedding(t, time_embed_dim)
        residuals = []

        x = input_layer(x)

        for layer in down_layers:
            for layer_block in layer:
                x = layer_block(x, t_emb, deterministic=not train)
            residuals.append(x)

        for layer in bottom_layers:
            x = layer(x, t_emb, deterministic=not train)

        for layer in up_layers:
            x = jnp.concatenate([x, residuals.pop()], axis=-1)
            for layer_block in layer:
                x = layer_block(x, t_emb, deterministic=not train)

        x = output_layer(x, deterministic=not train)

        return x


def main():

    batch_size = 2
    input_dim = (batch_size, 256, 256, 3)

    config = UNetConfig(
        input_dim=input_dim,
        dim_mults=(1, 2, 4, 8, 16),
        attention_mults=(32, 16, 8),
        time_embed_dim=16,
        num_resnet_blocks=2,
        num_norm_groups=4,
        dropout=0.0,
        num_init_channels=256,
        num_attention_heads=4,
        attention_head_dim=64,
        downsample_last=False,
        kernel_size=(3, 3)
    )

    x = jnp.ones(input_dim)
    t = jnp.ones(batch_size)

    model = UNet(config)
    random_key = random.PRNGKey(0)
    variables = model.init(random_key, x, t, train=True)

    model.apply(variables, x, t, train=True)


if __name__ == "__main__":
    main()
