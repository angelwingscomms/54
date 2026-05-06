import jax
import jax.numpy as jnp
from functools import partial
from .tkan_apply import tkan_apply


@partial(jax.jit)
def bce_loss(params, x, y):
    preds = tkan_apply(params, x)
    eps = 1e-8
    return -jnp.mean(y * jnp.log(preds + eps) + (1 - y) * jnp.log(1 - preds + eps))


def eval_loss(params, x, y, batch_size=128):
    total, count = 0.0, 0
    for i in range(0, len(x), batch_size):
        bx, by = x[i:i+batch_size], y[i:i+batch_size]
        total += float(bce_loss(params, bx, by))
        count += 1
    return total / count if count > 0 else 0.0