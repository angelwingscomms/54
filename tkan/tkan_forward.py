import jax.numpy as jnp
from .tkan_cell import tkan_cell


def tkan_fwd(params, x, hidden=100, sub=20):
    bs, seq, _ = x.shape
    h = jnp.zeros((bs, hidden))
    c = jnp.zeros((bs, hidden))
    sub_s = jnp.zeros((bs, sub))
    for t in range(seq):
        h, c, sub_s = tkan_cell(params, h, c, x[:, t, :], sub_s, hidden, sub)
    return h