import os
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import time
import optax
from jax2onnx import to_onnx

jax.default_backend = 'cpu'

DEFAULTS = {
    'symbol': 'BTC',
    'target_type': 'atr',
    'atr_multiplier': 2.0,
    'tp_multiplier': 2.0,
    'atr_period': 9,
    'threshold_pct': 1.0,
    'stop_loss_pct': 0.5,
    'n_ahead': 9,
}

def load_config():
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f) or {}
    for k, v in DEFAULTS.items():
        if k not in cfg:
            print(f'WARNING: {k} not specified, using default {v}')
    return {**DEFAULTS, **cfg}


def load_data(path='examples/data.csv', tp_pct=3.0, tolerance=0.2, horizon=24,
               target_type='atr', atr_multiplier=2.0, tp_multiplier=2.0, atr_period=9):
    df = pd.read_csv(path, index_col=0, parse_dates=True, encoding='utf-16')
    
    btc = df[['open', 'high', 'low', 'close']].copy()
    
    prev_close = btc['close'].shift(1)
    tr = np.maximum(
        btc['high'] - btc['low'],
        np.maximum(
            np.abs(btc['high'] - prev_close),
            np.abs(btc['low'] - prev_close)
        )
    )
    atr = tr.rolling(window=atr_period).mean()
    
    sl_pct = -tp_pct * tolerance
    
    X, y = [], []
    for i in range(45, len(btc) - horizon):
        X.append(btc.iloc[i - 45:i].values)
        
        close = btc.iloc[i]['close']
        
        if target_type == 'atr':
            atr_val = atr.iloc[i]
            sl_distance = atr_val * atr_multiplier
            sl_price = close - sl_distance
            tp_price = close + (sl_distance * tp_multiplier)
        else:
            tp_price = close * (1 + tp_pct / 100)
            sl_price = close * (1 + sl_pct / 100)
        
        tp_idx = sl_idx = None
        for j in range(1, horizon + 1):
            high = btc.iloc[i + j]['high']
            low = btc.iloc[i + j]['low']
            if tp_idx is None and high >= tp_price:
                tp_idx = j
            if sl_idx is None and low <= sl_price:
                sl_idx = j
            if tp_idx and sl_idx:
                break
        
        y.append(1.0 if tp_idx and (sl_idx is None or tp_idx < sl_idx) else 0.0)
    
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)


def normalize(xmin, xmax, X_tr, X_te, y_tr, y_te):
    X_tr = (X_tr - xmin) / (xmax - xmin + 1e-8)
    X_te = (X_te - xmin) / (xmax - xmin + 1e-8)
    return X_tr, X_te, y_tr, y_te


def save_norm_params(xmin, xmax):
    xmin = xmin.squeeze()
    xmax = xmax.squeeze()
    n = len(xmin)
    min_str = ", ".join(f"{v:.10g}" for v in xmin)
    max_str = ", ".join(f"{v:.10g}" for v in xmax)
    content = (
        f"const double NORM_MIN[{n}] = {{{min_str}}};\n"
        f"const double NORM_MAX[{n}] = {{{max_str}}};\n"
    )
    with open("norm_params.mqh", "w") as f:
        f.write(content)
    print(f"Saved norm_params.mqh ({n} features)")


def save_config(cfg):
    content = []
    mappings = [
        ('symbol', 'string', 'CFG_SYMBOL'),
        ('target_type', 'string', 'CFG_TARGET_TYPE'),
        ('atr_multiplier', 'double', 'CFG_ATR_MULTIPLIER'),
        ('tp_multiplier', 'double', 'CFG_TP_MULTIPLIER'),
        ('atr_period', 'int', 'CFG_ATR_PERIOD'),
        ('threshold_pct', 'double', 'CFG_THRESHOLD_PCT'),
        ('stop_loss_pct', 'double', 'CFG_TOLERANCE'),
        ('sequence_length', 'int', 'CFG_SEQUENCE_LENGTH'),
    ]
    for key, mql_type, name in mappings:
        v = cfg.get(key)
        if v is None:
            v = DEFAULTS.get(key)
        if mql_type == 'string':
            content.append(f'const string {name} = "{v}";')
        else:
            content.append(f'const {mql_type} {name} = {v};')
    
    with open('config.mqh', 'w') as f:
        f.write('\n'.join(content))
    print('Saved config.mqh')


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
    print('Loading config...', flush=True)
    cfg = load_config()
    
    print('Loading data...', flush=True)
    X, y = load_data(
        path=cfg['data_path'],
        tp_pct=cfg['threshold_pct'],
        tolerance=cfg['stop_loss_pct'],
        horizon=cfg['n_ahead'],
        target_type=cfg['target_type'],
        atr_multiplier=cfg['atr_multiplier'],
        tp_multiplier=cfg['tp_multiplier'],
        atr_period=cfg['atr_period']
    )
    print(f"Loaded X: {X.shape}", flush=True)
    
    sep = int(len(X) * 0.8)
    X_tr, X_te = X[:sep], X[sep:]
    y_tr, y_te = y[:sep], y[sep:]
    
    xmin, xmax = X_tr.min(axis=(0, 1), keepdims=True), X_tr.max(axis=(0, 1), keepdims=True)
    X_tr, X_te, y_tr, y_te = normalize(xmin, xmax, X_tr, X_te, y_tr, y_te)
    
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
    print("\n=== TKAN ===")
    key, k = jax.random.split(key)
    tkan_p = init_tkan(input_dim, hidden, sub, k)
    print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(tkan_p))}")
    
    opt = optax.adam(1e-3)
    opt_st = opt.init(tkan_p)
    
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

    start = time.time()
    train_losses, val_losses = [], []
    for ep in range(27):
        idx = jax.random.permutation(jax.random.key(ep), len(X_tr))
        ep_loss = 0
        for i in range(0, len(X_tr), 128):
            b_idx = idx[i:i+128]
            bx, by = X_tr[b_idx], y_tr[b_idx]
            l, g = jax.value_and_grad(bce_loss)(tkan_p, bx, by)
            u, opt_st = opt.update(g, opt_st)
            tkan_p = optax.apply_updates(tkan_p, u)
            ep_loss += l
        num_batches = len(range(0, len(X_tr), 128))
        train_loss = float(ep_loss) / num_batches
        val_loss = eval_loss(tkan_p, X_te, y_te)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (ep+1) % 2 == 0:
            print(f"  Epoch {ep+1}: train={train_loss:.4f}  val={val_loss:.4f}")
    
    tkan_time = time.time() - start
    preds = tkan_apply(tkan_p, X_te)
    acc = jnp.mean((preds > 0.5) == y_te)
    print(f"Time: {tkan_time:.1f}s, Accuracy: {acc:.4f}")
    
    print("\n" + "="*48)
    print("SUMMARY")
    print("="*48)
    print(f"{'Epoch':>6} | {'Train':>8} | {'Val':>8}")
    print("-"*48)
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        print(f"{i+1:>6} | {tl:>8.4f} | {vl:>8.4f}")
    print("="*48)
    print(f"TKAN time: {tkan_time:.1f}s  Final val_acc: {acc:.4f}")

    save_norm_params(xmin, xmax)
    save_config(cfg)

    print("\nExporting model to ONNX...")
    def make_apply_fn(params):
        def apply_fn(x, hidden=100, sub=20):
            return jax.nn.sigmoid(jnp.dot(tkan_fwd(params, x, hidden), params['dense_w']) + params['dense_b'])
        return apply_fn

    result = to_onnx(
        make_apply_fn(tkan_p),
        inputs=[jax.ShapeDtypeStruct((1, 45, 4), jnp.float32)],
        model_name='TKAN',
        return_mode='file',
        output_path='model.onnx'
    )
    print(f"Model saved to: model.onnx")

if __name__ == '__main__':
    main()