import jax
import jax.numpy as jnp


def init_tkan(input_dim, hidden, sub, key):
    k = jax.random.split(key, 6)
    return {
        'wx': jax.random.normal(k[0], (input_dim, hidden * 3)) * 0.3,
        'uh': jax.random.normal(k[1], (hidden, hidden * 3)) * 0.3,
        'bias': jnp.zeros((hidden * 3,)),
        'sub_wx': jax.random.normal(k[2], (input_dim, sub)) * 0.2,
        'sub_wh': jax.random.normal(k[3], (sub, sub)) * 0.2,
        'sub_k': jax.random.normal(k[4], (sub * 2,)) * 0.2,
        'agg_w': jax.random.normal(k[5], (sub, hidden)) * 0.3,
        'agg_b': jnp.zeros((hidden,)),
        'dense_w': jax.random.normal(k[0], (hidden, 1)) * 0.3,
        'dense_b': jnp.zeros((1,)),
    }