import jax.numpy as jnp

def cast_dim(x, n_dims):
    dim_diff = n_dims - x.ndim
    extra_dim_indices = [-i for i in range(1, dim_diff+1)]
    return jnp.expand_dims(x, extra_dim_indices)