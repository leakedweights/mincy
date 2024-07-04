import jax
import jax.numpy as jnp
from jax import random

from .karras_utils import cin, cout, cskip, cnoise
from ..models.utils import cast_dim


def pseudo_huber_loss(x: jax.Array, y: jax.Array, c_data: float):
    loss = (x - y) ** 2
    loss = jnp.sqrt(loss + c_data**2) - c_data
    return loss


def consistency_fn(xt, sigma, sigma_data, sigma_min, apply_fn, params):
    input = cast_dim(cin(sigma, sigma_data), xt.ndim) * xt
    output = apply_fn(params, input * xt, cnoise(sigma))
    consistency_out = cout(sigma, sigma_data, sigma_min) * \
        output + cskip(sigma, sigma_data, sigma_min) * xt

    return output, consistency_out


def sample_single_step(key, denoising_fn, shape, sigma_data, sigma_min, sigma_max):
    xT = random.normal(key, shape) * sigma_max
    sigmas = sigma_max * jnp.ones(shape[:1])
    sigmas = cast_dim(sigmas, xT.ndim)
    _, sample = consistency_fn(xT, sigmas, sigma_data, sigma_min, denoising_fn)
    sample = jnp.clip(sample, -1, 1)
    return sample
