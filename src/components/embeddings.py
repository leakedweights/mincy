import jax
import jax.numpy as jnp
import flax.linen as nn


def sinusoidal_emb(timesteps: jax.Array, embedding_dim: int):
    timesteps = timesteps * 1e3
    half_dim = embedding_dim // 2
    emb_scale = jnp.log(1e4) / (half_dim - 1)

    emb = jnp.arange(half_dim) * -emb_scale
    emb = jnp.exp(emb)
    emb = emb[None, :] * timesteps[:, None]

    sin_emb = jnp.sin(emb)
    cos_emb = jnp.cos(emb)
    embedding = jnp.concatenate([sin_emb, cos_emb], axis=-1)

    if embedding_dim % 2 == 1:
        padding = ((0, 0), (0, 0), (0, 1))
        embedding = jnp.pad(embedding, padding, mode='constant')

    return embedding


class FourierEmbedding(nn.Module):
    """from https://github.com/openai/consistency_models_cifar10/"""
    embedding_size: int = 256
    scale: float = 1.0

    @nn.compact
    def __call__(self, x):
        W = self.param(
            "W", jax.nn.initializers.normal(
                stddev=self.scale), (self.embedding_size,)
        )
        W = jax.lax.stop_gradient(W)
        x_proj = x[:, None] * W[None, :] * 2 * jnp.pi
        return jnp.concatenate([jnp.sin(x_proj), jnp.cos(x_proj)], axis=-1)
