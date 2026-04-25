import jax
import jax.numpy as jnp
from functools import partial
from .tkan_cell import tkan_cell


@partial(jax.jit, static_argnames=['hidden', 'sub'])
def tkan_fwd(params, x, hidden=100, sub=20):
    bs, seq, _ = x.shape
    
    def step(carry, t):
        h, c, sub_s = carry
        h, c, sub_s = tkan_cell(params, h, c, x[:, t, :], sub_s, hidden, sub)
        return (h, c, sub_s), h
    
    h0 = jnp.zeros((bs, hidden))
    c0 = jnp.zeros((bs, hidden))
    sub0 = jnp.zeros((bs, sub))
    _, hs = jax.lax.scan(step, (h0, c0, sub0), jnp.arange(seq))
    return hs[-1]