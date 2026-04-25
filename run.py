import os
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import optax

jax.default_backend = 'cpu'


def load_data(path='examples/data.parquet', assets=None):
    df = pd.read_parquet(path)
    df = df[(df.index >= pd.Timestamp('2020-01-01')) & (df.index < pd.Timestamp('2023-01-01'))]
    if assets is None:
        assets = ['BTC', 'ETH', 'ADA', 'XMR', 'EOS', 'MATIC', 'TRX', 'FTM', 'BNB', 'XLM', 'ENJ', 'CHZ', 'BUSD', 'ATOM', 'LINK', 'ETC', 'XRP', 'BCH', 'LTC']
    df = df[[c for c in df.columns if 'quote asset volume' in c and any(a in c for a in assets)]]
    
    X, y = [], []
    for i in range(45, len(df) - 1):
        X.append(df.iloc[i - 45:i].values)
        y.append(1.0 if df.iloc[i + 1, 0] > df.iloc[i, 0] else 0.0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def normalize(X_tr, X_te, y_tr, y_te):
    xmin, xmax = X_tr.min(axis=(0, 1), keepdims=True), X_tr.max(axis=(0, 1), keepdims=True)
    X_tr = (X_tr - xmin) / (xmax - xmin + 1e-8)
    X_te = (X_te - xmin) / (xmax - xmin + 1e-8)
    return X_tr, X_te, y_tr, y_te


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


def tkan_cell(params, h, c, x, sub_s, hidden=100, sub=20):
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
    return h_new, c_new, new_sub


def tkan_fwd(params, x, hidden=100, sub=20):
    bs, seq, _ = x.shape
    h = jnp.zeros((bs, hidden))
    c = jnp.zeros((bs, hidden))
    sub_s = jnp.zeros((bs, sub))
    for t in range(seq):
        h, c, sub_s = tkan_cell(params, h, c, x[:, t, :], sub_s, hidden, sub)
    return h


def tkan_apply(params, x, hidden=100):
    return jax.nn.sigmoid(jnp.dot(tkan_fwd(params, x, hidden), params['dense_w']) + params['dense_b'])





def main():
    print("Loading data...", flush=True)
    X, y = load_data()
    print(f"Loaded X: {X.shape}", flush=True)
    
    sep = int(len(X) * 0.8)
    X_tr, X_te = X[:sep], X[sep:]
    y_tr, y_te = y[:sep], y[sep:]
    
    X_tr, X_te, y_tr, y_te = normalize(X_tr, X_te, y_tr, y_te)
    
    X_tr = jnp.array(X_tr)
    y_tr = jnp.array(y_tr)
    X_te = jnp.array(X_te)
    y_te = jnp.array(y_te)
    
    # jax.config.update("jax_disable_jit", True)
    
    print(f"X_train: {X_tr.shape}, y_train: {y_tr.shape}")
    print(f"X_test: {X_te.shape}, y_test: {y_te.shape}")
    
    hidden, sub = 100, 20
    input_dim = X_tr.shape[-1]
    
    key = jax.random.key(42)
    n_train = len(X_tr) // 10
    X_tr_small = X_tr[:n_train]
    y_tr_small = y_tr[:n_train]
    
    print("\n=== TKAN ===")
    key, k = jax.random.split(key)
    tkan_p = init_tkan(input_dim, hidden, sub, k)
    print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(tkan_p))}")
    
    opt = optax.adam(1e-3)
    opt_st = opt.init(tkan_p)
    
    start = time.time()
    for ep in range(10):
        idx = jax.random.permutation(jax.random.key(ep), len(X_tr))
        ep_loss = 0
        for i in range(0, len(X_tr), 128):
            b_idx = idx[i:i+128]
            bx, by = X_tr[b_idx], y_tr[b_idx]
            
            def loss_fn(p):
                preds = tkan_apply(p, bx)
                eps = 1e-8
                loss = -jnp.mean(by * jnp.log(preds + eps) + (1 - by) * jnp.log(1 - preds + eps))
                return loss
            
            l, g = jax.value_and_grad(loss_fn)(tkan_p)
            u, opt_st = opt.update(g, opt_st)
            tkan_p = optax.apply_updates(tkan_p, u)
            ep_loss += l
        
        if (ep+1) % 2 == 0:
            print(f"  Epoch {ep+1}: loss = {ep_loss:.4f}")
    
    tkan_time = time.time() - start
    preds = tkan_apply(tkan_p, X_te)
    acc = jnp.mean((preds > 0.5) == y_te)
    print(f"Time: {tkan_time:.1f}s, Accuracy: {acc:.4f}")
    
    print("\n" + "="*40)
    print("SUMMARY")
    print("="*40)
    print(f"TKAN: {tkan_time:.1f}s")
    print("="*40)


if __name__ == '__main__':
    main()