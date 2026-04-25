import jax
import jax.numpy as jnp
from .tkan_forward import tkan_fwd


def tkan_apply(params, x, hidden=100):
    return jax.nn.sigmoid(jnp.dot(tkan_fwd(params, x, hidden), params['dense_w']) + params['dense_b'])