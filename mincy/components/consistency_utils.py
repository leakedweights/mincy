import jax
import jax.numpy as jnp
from jax import random
from scipy.special import erf


from .karras_utils import cin, cout, cskip, cnoise
from ..models.utils import cast_dim


def discretize(step, s0, s1, max_steps):
    k_prime = jnp.floor(max_steps / (jnp.log2(jnp.floor(s1 / s0)) + 1))
    N = s0 * jnp.pow(2, jnp.floor(step / k_prime))
    N = jnp.min(jnp.array([N, s1]))
    return N + 1


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


def pseudo_huber_loss(x: jax.Array, y: jax.Array, c_data: float):
    loss = (x - y) ** 2
    loss = jnp.sqrt(loss + c_data**2) - c_data
    return loss


def consistency_fn(xt, sigma, sigma_data, sigma_min, apply_fn, params):
    input = cast_dim(cin(sigma, sigma_data), xt.ndim) * xt
    output = apply_fn(params, input * xt, cnoise(sigma))
    consistency_out = cast_dim(cout(sigma, sigma_data, sigma_min), xt.ndim) * \
        output + cast_dim(cskip(sigma, sigma_data, sigma_min), xt.ndim) * xt

    return output, consistency_out


def sample_single_step(key, denoising_fn, denoising_params, shape, sigma_data, sigma_min, sigma_max):
    xT = random.normal(key, shape) * sigma_max
    sigmas = sigma_max * jnp.ones(shape[:1])
    sigmas = cast_dim(sigmas, xT.ndim)
    _, sample = consistency_fn(xT, sigmas, sigma_data, sigma_min, denoising_fn, denoising_params)
    sample = jnp.clip(sample, -1, 1)
    return sample
