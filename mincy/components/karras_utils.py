import jax
import jax.numpy as jnp


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
