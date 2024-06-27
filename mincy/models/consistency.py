import jax
import jax.numpy as jnp
from jax import random

from functools import partial

from ..components.solvers import euler
from .utils import cast_dim


def pseudo_huber_loss(x: jax.Array, y: jax.Array, c_data: float):
    loss = (x - y) ** 2
    loss = jnp.sqrt(loss + c_data**2) - c_data
    return loss


def consistency_fn(xt, sigma, sigma_data, sigma_min, apply_fn, params):
    cin = jnp.pow((sigma**2 + sigma_data**2), -0.5)
    cin = cast_dim(cin, xt.ndim)

    cout = ((sigma - sigma_min) * sigma_data) / \
        jnp.sqrt((sigma**2 + sigma_data**2))
    cout = cast_dim(cout, xt.ndim)

    cskip = sigma**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    cskip = cast_dim(cskip, xt.ndim)

    scaled_sigma = 0.25 * jnp.log(sigma + 1e-44)

    out = apply_fn(params, cin * xt, scaled_sigma)
    consistency_out = cout * out + cskip * xt

    return out, consistency_out


def training_consistency(t1, t2, x0, noise, apply_fn, params, sigma_data, sigma_min):

    t1_noise_dim = cast_dim(t1, noise.ndim)
    t2_noise_dim = cast_dim(t2, noise.ndim)

    xt1 = x0 + t1_noise_dim * noise
    xt2 = euler(xt1, t1_noise_dim, t2_noise_dim, x0)

    _, xt1_consistency = consistency_fn(
        xt1, t1, sigma_data, sigma_min, apply_fn, params)
    _, xt2_consistency = consistency_fn(
        xt2, t2, sigma_data, sigma_min, apply_fn, params)

    return xt1_consistency, xt2_consistency


def sample_single_step(key, denoising_fn, shape, sigma_data, sigma_min, sigma_max):
    xT = random.normal(key, shape) * sigma_max
    sigmas = sigma_max * jnp.ones(shape[:1])
    sigmas = cast_dim(sigmas, xT.ndim)
    _, sample = consistency_fn(xT, sigmas, sigma_data, sigma_min, denoising_fn)
    sample = jnp.clip(sample, -1, 1)
    return sample
