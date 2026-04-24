import jax
import jax.numpy as jnp
from jax import lax


def init_tkan(input_dim, hidden, sub, key):
    k = jax.random.split(key, 7)
    return {
        'wx': jax.random.normal(k[0], (input_dim, hidden * 3)) * 0.3,
        'uh': jax.random.normal(k[1], (hidden, hidden * 3)) * 0.3,
        'bias': jnp.concatenate([jnp.zeros((hidden,)), jnp.ones((hidden,)), jnp.zeros((hidden,))]),
        'sub_wx': jax.random.normal(k[2], (input_dim, sub)) * 0.2,
        'sub_wh': jax.random.normal(k[3], (sub, sub)) * 0.2,
        'sub_k': jax.random.normal(k[4], (sub * 2,)) * 0.2,
        'agg_w': jax.random.normal(k[5], (sub, hidden)) * 0.3,
        'agg_b': jnp.zeros((hidden,)),
        'dense_w': jax.random.normal(k[6], (hidden, 1)) * 0.3,
        'dense_b': jnp.zeros((1,)),
    }


def _tkan_cell(carry, x, params, hidden, sub):
    h, c, sub_s = carry
    gates = jnp.dot(x, params['wx']) + jnp.dot(h, params['uh']) + params['bias']
    i = jax.nn.sigmoid(gates[:, :hidden])
    f = jax.nn.sigmoid(gates[:, hidden:hidden*2])
    cg = jnp.tanh(gates[:, hidden*2:])
    c_new = f * c + i * cg
    sub_o = jnp.tanh(jnp.dot(x, params['sub_wx']) + jnp.dot(sub_s, params['sub_wh']))
    kh, kx = params['sub_k'][:sub], params['sub_k'][sub:]
    new_sub = kh * sub_o + kx * sub_s
    agg = jnp.dot(sub_o, params['agg_w']) + params['agg_b']
    o = jax.nn.sigmoid(agg)
    h_new = o * jnp.tanh(c_new)
    return (h_new, c_new, new_sub), h_new


@jax.jit
def tkan_fwd(params, x):
    bs = x.shape[0]
    sub = params['sub_wx'].shape[1]
    hidden = params['wx'].shape[1] // 3
    h0 = jnp.zeros((bs, hidden))
    c0 = jnp.zeros((bs, hidden))
    s0 = jnp.zeros((bs, sub))
    x_t = jnp.swapaxes(x, 0, 1)
    def step(carry, xt):
        return _tkan_cell(carry, xt, params, hidden, sub)
    final, _ = lax.scan(step, (h0, c0, s0), x_t)
    return final[0]


def tkan_apply(params, x, use_sigmoid=False):
    out = jnp.dot(tkan_fwd(params, x), params['dense_w']) + params['dense_b']
    return jax.nn.sigmoid(out) if use_sigmoid else out


def binary_crossentropy(pred, target, eps=1e-5):
    pred = jnp.clip(pred, eps, 1 - eps)
    return jnp.mean(-target * jnp.log(pred) - (1 - target) * jnp.log(1 - pred))


def compute_accuracy(pred, target, threshold=0.5):
    return jnp.mean((pred > threshold).astype(jnp.float32) == target)


def make_apply_fn(params, use_sigmoid=False):
    def apply_fn(x):
        return tkan_apply(params, x, use_sigmoid=use_sigmoid)
    return apply_fn
