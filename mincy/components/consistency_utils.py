import jax
import jax.numpy as jnp
from jax import random
from scipy.special import erf

from ..utils import cast_dim

from typing import Callable, Any, Iterable


def discretize(step: jax.Array, s0: int, s1: int, max_steps: int) -> jax.Array:
    k_prime = jnp.floor(max_steps / (jnp.log2(jnp.floor(s1 / s0)) + 1))
    N = s0 * jnp.pow(2, jnp.floor(step / k_prime))
    N = jnp.min(jnp.array([N, s1]))
    return N + 1


def get_boundaries(N: jax.Array, sigma_min: float, sigma_max: float, rho: float) -> jax.Array:
    discretized_steps = jnp.arange(1, N)
    levels = sigma_min ** (1/rho) + (discretized_steps / (N-1)) * \
        (sigma_max ** (1/rho) - sigma_min ** (1/rho))
    levels = levels ** (rho)
    return levels


def cskip(sigma: jax.Array, sigma_data: float, sigma_min: float = 0.0) -> jax.Array:
    return sigma_data ** 2 / ((sigma - sigma_min) ** 2 + sigma_data ** 2)


def cout(sigma: jax.Array, sigma_data: float, sigma_min: float = 0.0) -> jax.Array:
    return (sigma - sigma_min) * sigma_data / jnp.sqrt(sigma ** 2 + sigma_data ** 2)


def cin(sigma: jax.Array, sigma_data: float) -> jax.Array:
    return 1 / jnp.sqrt(sigma ** 2 + sigma_data ** 2)


def cnoise(sigma: jax.Array) -> jax.Array:
    return 0.25 * jnp.log(sigma + 1e-44)


def sample_timesteps(key: Any, noise_levels: jax.Array, shape: Iterable, p_mean: float, p_std: float) -> tuple[jax.Array, jax.Array]:
    erf_1 = erf((jnp.log(noise_levels[:-1]) - p_mean) / (jnp.sqrt(2) * p_std))
    erf_2 = erf((jnp.log(noise_levels[1:]) - p_mean) / (jnp.sqrt(2) * p_std))
    p_sigmas = erf_2 - erf_1
    p_sigmas = p_sigmas / jnp.sum(p_sigmas)
    indices = random.choice(
        key, len(noise_levels[:-1]), shape, replace=True, p=p_sigmas)

    t1 = noise_levels[indices]
    t2 = noise_levels[indices + 1]
    return t1, t2


def pseudo_huber_loss(x: jax.Array, y: jax.Array, c_data: float):
    loss = (x - y) ** 2
    loss = jnp.sqrt(loss + c_data**2) - c_data
    return loss


def consistency_fn(xt: jax.Array, y: jax.Array, sigma: jax.Array, sigma_data: float,
                   sigma_min: float, apply_fn: Callable, params: Any) -> tuple[jax.Array, jax.Array]:
    input = cast_dim(cin(sigma, sigma_data), xt.ndim) * xt
    output = apply_fn(params, input * xt, y, cnoise(sigma))
    consistency_out = cast_dim(cout(sigma, sigma_data, sigma_min), xt.ndim) * \
        output + cast_dim(cskip(sigma, sigma_data, sigma_min), xt.ndim) * xt

    return output, consistency_out


def sample_single_step(key: Any, denoising_fn: Callable, denoising_params: Any,
                       shape: Iterable, sigma_data: float, sigma_min: float,
                       sigma_max: float, num_classes: int, classes=None) -> jax.Array:
    xT = random.normal(key, shape) * sigma_max
    sigmas = sigma_max * jnp.ones(shape[:1])

    if classes is None:
        generation_classes = [i % num_classes for i in range(shape[0])]

    _, sample = consistency_fn(
        xT, generation_classes, sigmas, sigma_data,
        sigma_min, denoising_fn, denoising_params)

    sample = jnp.clip(sample, -1, 1)
    return sample
