from jax import random
import jax.numpy as jnp

from components.utils import cast_dim
from components.consistency import consistency_fn

def sample_single_step(key, denoising_fn, shape, sigma_data, sigma_min, sigma_max):
    xT = random.normal(key, shape) * sigma_max
    sigmas = sigma_max * jnp.ones(shape[:1])
    sigmas = cast_dim(sigmas, xT.ndim)
    _, sample = consistency_fn(xT, sigmas, sigma_data, sigma_min, denoising_fn)
    sample = jnp.clip(sample, -1, 1)
    return sample