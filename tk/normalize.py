import jax.numpy as jnp

def normalize(xmean, xstd, *arrays):
    return tuple(jnp.clip((arr - xmean) / (xstd + 1e-7), -3.0, 3.0) if getattr(arr, 'ndim', 0) == 3 else arr for arr in arrays)
