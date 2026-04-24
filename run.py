import os
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import optax
from jax import lax
from jax2onnx import to_onnx

print("Loading libraries...", flush=True)
jax.default_backend = 'cpu'

df = pd.read_parquet('examples/data.parquet')
df = df[(df.index >= pd.Timestamp('2020-01-01')) & (df.index < pd.Timestamp('2023-01-01'))]
assets = ['BTC', 'ETH', 'ADA', 'XMR', 'EOS', 'MATIC', 'TRX', 'FTM', 'BNB', 'XLM', 'ENJ', 'CHZ', 'BUSD', 'ATOM', 'LINK', 'ETC', 'XRP', 'BCH', 'LTC']
df = df[[c for c in df.columns if 'quote asset volume' in c and any(a in c for a in assets)]]

print(f"Data shape: {df.shape}", flush=True)

X = np.array([df.iloc[i - 45:i].values for i in range(45, len(df) - 1)], dtype=np.float32)
y = np.array([df.iloc[i:i + 1, 0:1].values for i in range(45, len(df) - 1)], dtype=np.float32)

sep = int(len(X) * 0.8)
X_tr, X_te = X[:sep], X[sep:]
y_tr, y_te = y[:sep], y[sep:]

xmin, xmax = X_tr.min(), X_tr.max()
X_tr = (X_tr - xmin) / (xmax - xmin + 1e-8)
X_te = (X_te - xmin) / (xmax - xmin + 1e-8)

ymin, ymax = y_tr.min(), y_tr.max()
y_tr = (y_tr - ymin) / (ymax - ymin + 1e-8)
y_te = (y_te - ymin) / (ymax - ymin + 1e-8)
y_tr = y_tr.reshape(len(y_tr), -1)
y_te = y_te.reshape(len(y_te), -1)

X_tr = jnp.array(X_tr)
y_tr = jnp.array(y_tr)
X_te = jnp.array(X_te)
y_te = jnp.array(y_te)

print(f"Train: {X_tr.shape}, Test: {X_te.shape}", flush=True)

hidden, sub = 100, 20
input_dim = X_tr.shape[-1]

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

def tkan_cell(carry, x, params, hidden=100, sub=20):
    h, c, sub_s = carry
    gates = jnp.dot(x, params['wx']) + jnp.dot(h, params['uh']) + params['bias']
    i = jax.nn.sigmoid(gates[:, :hidden])
    f = jax.nn.sigmoid(gates[:, hidden:hidden*2])
    cg = jnp.tanh(gates[:, hidden*2:])
    c_new = f * c + i * cg
    sub_o = jnp.tanh(jnp.dot(x, params['sub_wx']) + jnp.dot(sub_s, params['sub_wh']))
    kh = params['sub_k'][:sub]
    kx = params['sub_k'][sub:]
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
    sub_s0 = jnp.zeros((bs, sub))
    
    x_t = jnp.swapaxes(x, 0, 1)
    
    def step(carry, xt):
        return tkan_cell(carry, xt, params, hidden, sub)
    
    final, outputs = lax.scan(step, (h0, c0, sub_s0), x_t)
    return outputs[-1, :]

def tkan_apply(params, x):
    return jnp.dot(tkan_fwd(params, x), params['dense_w']) + params['dense_b']

def make_apply(params):
    def apply_fn(x):
        out = tkan_fwd(params, x)
        return jnp.dot(out, params['dense_w']) + params['dense_b']
    return apply_fn

def init_gru(input_dim, hidden, key):
    k = jax.random.split(key, 3)
    sc = jnp.sqrt(2.0 / (input_dim + hidden))
    return {
        'Wr': jax.random.normal(k[0], (input_dim, hidden)) * sc,
        'Uz': jax.random.normal(k[1], (hidden, hidden)) * sc,
        'b': jnp.zeros((hidden,)),
        'dense_w': jax.random.normal(k[2], (hidden, 1)) * jnp.sqrt(2.0 / (hidden + 1)),
        'dense_b': jnp.zeros((1,)),
    }

def gru_cell(h, x, params):
    z = jax.nn.sigmoid(jnp.dot(x, params['Wr']) + jnp.dot(h, params['Uz']) + params['b'])
    h_cand = jnp.tanh(jnp.dot(x, params['Wr']) + jnp.dot(z * h, params['Uz']) + params['b'])
    return (1 - z) * h + z * h_cand

@jax.jit
def gru_fwd(params, x):
    bs = x.shape[0]
    h0 = jnp.zeros((bs, params['Wr'].shape[1]))
    
    x_t = jnp.swapaxes(x, 0, 1)
    
    def step(h, xt):
        return gru_cell(h, xt, params), None
    
    final, _ = lax.scan(step, h0, x_t)
    return final[-1]

def gru_apply(params, x):
    return jnp.dot(gru_fwd(params, x), params['dense_w']) + params['dense_b']

key = jax.random.key(42)

print("\n=== TKAN ===", flush=True)
key, k = jax.random.split(key)
tkan_p = init_tkan(input_dim, hidden, sub, k)
print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(tkan_p))}", flush=True)

opt = optax.adam(1e-3)
opt_st = opt.init(tkan_p)

start = time.time()
for ep in range(3):
    idx = jax.random.permutation(jax.random.key(ep), len(X_tr))
    ep_loss = 0
    for i in range(0, min(len(X_tr), 2560), 256):
        b_idx = idx[i:i+256]
        bx, by = X_tr[b_idx], y_tr[b_idx]
        
        def loss_fn(p):
            return jnp.mean((tkan_apply(p, bx) - by) ** 2)
        
        l, g = jax.value_and_grad(loss_fn)(tkan_p)
        u, opt_st = opt.update(g, opt_st)
        tkan_p = optax.apply_updates(tkan_p, u)
        ep_loss += l
    
    print(f"  Epoch {ep+1}: loss = {ep_loss:.4f}", flush=True)

t_time = time.time() - start
preds = tkan_apply(tkan_p, X_te)
rmse = jnp.sqrt(jnp.mean((y_te - preds) ** 2))
print(f"Time: {t_time:.1f}s, RMSE: {rmse:.4f}", flush=True)

print("Exporting to ONNX...", flush=True)
result = to_onnx(
    make_apply(tkan_p),
    inputs=[jax.ShapeDtypeStruct((1, 45, 19), jnp.float32)],
    model_name='TKAN',
    return_mode='file',
    output_path='model.onnx'
)
print(f"Saved model.onnx: {result}", flush=True)