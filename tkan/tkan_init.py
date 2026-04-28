import jax
import jax.numpy as jnp
import math


def init_tkan(input_dim, hidden, sub, key):
    k = jax.random.split(key, 7)
    scale_lstm = 1.0 / math.sqrt(input_dim)
    scale_hidden = 1.0 / math.sqrt(hidden)
    scale_sub = 1.0 / math.sqrt(sub)
    return {
        'wx': jax.random.normal(k[0], (input_dim, hidden * 3)) * scale_lstm,
        'uh': jax.random.normal(k[1], (hidden, hidden * 3)) * scale_hidden,
        'bias': jnp.zeros((hidden * 3,)),
        'sub_wx': jax.random.normal(k[2], (input_dim, sub)) * scale_lstm,
        'sub_wh': jax.random.normal(k[3], (sub, sub)) * scale_sub,
        'sub_k': jax.random.normal(k[4], (sub * 2,)) * scale_sub,
        'agg_w': jax.random.normal(k[5], (sub + hidden, hidden)) * scale_sub,
        'agg_b': jnp.zeros((hidden,)),
        'dense_w': jax.random.normal(k[6], (hidden, 1)) * scale_hidden,
        'dense_b': jnp.zeros((1,)),
    }
