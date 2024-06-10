import jax
import jax.numpy as jnp
from jax import random
from scipy.special import erf

from functools import partial


@partial(jax.jit, static_argnums=(1, 2, 3))
def discretize(step, s0, s1, max_steps):
    k_prime = jnp.floor(max_steps / (jnp.log2(jnp.floor(s1 / s0)) + 1))
    N = s0 * jnp.pow(2, jnp.floor(step / k_prime))
    N = jnp.min(jnp.array([N, s1])).astype(int)
    return N + 1


@partial(jax.jit, static_argnums=(2, 3, 4))
def karras_levels(i, N, sigma_min, sigma_max, rho):
    levels = sigma_max ** (1/rho) + (i / (N-1)) * \
        (sigma_min ** (1/rho) - sigma_max ** (1/rho))
    levels = levels ** (rho)
    return levels


@partial(jax.jit, static_argnums=(2, 3, 4))
def sample_timesteps(key, noise_levels, shape, p_mean, p_std):
    erf_1 = erf((jnp.log(noise_levels[:-1]) - p_mean) / (jnp.sqrt(2) * p_std))
    erf_2 = erf((jnp.log(noise_levels[1:]) - p_mean) / (jnp.sqrt(2) * p_std))
    p_sigmas = erf_2 - erf_1
    p_sigmas = p_sigmas / jnp.sum(p_sigmas)
    indices = random.choice(
        key, len(noise_levels[:-1]), shape, replace=True, p=p_sigmas)

    t1 = noise_levels[indices]
    t2 = noise_levels[indices + 1]
    return t1, t2
