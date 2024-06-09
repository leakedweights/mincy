import jax


@jax.jit
def euler(xt, t1, t2, x0):
    d = (xt - x0) / t1
    xt2 = xt * d * (t2 - t1)
    return xt2


@jax.jit
def heun(xt, t1, t2, x0):
    raise NotImplementedError("This solver is not implemented!")
