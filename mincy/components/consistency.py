import jax
from jax import random
import jax.numpy as jnp

from utils import cast_dim
from functools import partial


@partial(jax.jit, static_argnums=(2,))
def pseudo_huber_loss(x: jax.Array, y: jax.Array, c_data: float):
    loss = (x - y) ** 2
    loss = jnp.sqrt(loss + c_data**2) - c_data
    return loss


@jax.jit
def euler(xt, t1, t2, x0):
    d = (xt - x0) / t1
    xt2 = xt * d * (t2 - t1)
    return xt2


@partial(jax.jit, static_argnums=(2, 3, 4))
def consistency_fn(xt, sigma, sigma_data, sigma_min, denoising_fn):
    cin = jnp.pow((sigma**2 + sigma_data**2), -0.5)
    cin = cast_dim(cin, xt.ndim)

    cout = ((sigma - sigma_min) * sigma_data) / \
        jnp.sqrt((sigma**2 + sigma_data**2))
    cout = cast_dim(cout, xt.ndim)

    cskip = sigma**2 / ((sigma - sigma_min)**2 + sigma_data**2)
    cskip = cast_dim(cskip, xt.ndim)

    scaled_sigma = 1e4 * 0.25 * jnp.log(sigma + 1e-44)
    out = denoising_fn(cin * xt, scaled_sigma)
    consistency_out = cout * out + cskip * xt
    return out, consistency_out


@partial(jax.jit, static_argnums=(5, 6))
def denoise_fn(t1, t2, x0, noise, denoising_fn, sigma_data, sigma_min):
    xt1 = x0 + t1 * noise
    xt2 = euler(xt1, t1, t2, x0)

    _, xt1_consistency = consistency_fn(
        xt1, t1, sigma_data, sigma_min, denoising_fn)
    _, xt2_consistency = consistency_fn(
        xt2, t2, sigma_data, sigma_min, denoising_fn)

    return xt1_consistency, xt2_consistency
