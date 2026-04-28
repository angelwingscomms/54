#!/usr/bin/env python3
import os
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

import argparse
import yaml
import time
import subprocess
from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console

jax.default_backend = 'cpu'

console = Console()


def load_config(path='r.yaml'):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault('hidden_size', 100)
    cfg.setdefault('sub_dim', 20)
    cfg.setdefault('epochs', 100)
    cfg.setdefault('batch_size', 128)
    cfg.setdefault('learning_rate', 0.001)
    cfg.setdefault('seed', 42)
    return cfg


def load_csv(path):
    if not path.endswith('.csv'):
        path += '.csv'
    df = pd.read_csv(path, encoding='utf-16', parse_dates=['datetime'], date_format='%Y-%m-%d %H-%M')
    if 'symbol' not in df.columns:
        return df.set_index('datetime').sort_index()
    wide = df.pivot(index='datetime', columns='symbol', values=['open', 'high', 'low', 'close', 'tick_volume'])
    wide.columns = [f'{symbol}_{field}' for symbol, field in wide.columns]
    return wide.sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, default='', help='Name prefix for model folder')
    args = parser.parse_args()

    cfg = load_config()
    prefix = args.name + '-' if args.name else ''
    model_dir = Path(f"models/{prefix}{datetime.now().strftime('%d%m-%H%M%S')}")
    model_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 60)
    print("# r.py - TKAN REGRESSION")
    print("#" * 60 + "\n")

    print(f"  Model output directory: {model_dir}")

    print("\n" + "=" * 50)
    print("LOADING DATA")
    print("=" * 50)
    print(f"  Path: {cfg['data_path']}")
    print(f"  Parameters:")
    print(f"    - symbol: {cfg['symbol']}")
    print(f"    - sequence_length: {cfg['sequence_length']}")
    print(f"    - n_ahead: {cfg['n_ahead']}")
    print(f"    - train_test_split: {cfg['train_test_split']}")
    print(f"    - feature_symbols: {list(cfg.get('feature_symbols', {}).keys())}")

    print(f"\n  Loading CSV file...")
    df = load_csv(f'./data/{cfg["data_path"]}')
    print(f"  CSV loaded: {len(df)} rows")

    target = cfg['symbol']
    feat_syms = [s for s in cfg.get('feature_symbols', {}) if cfg['feature_symbols'].get(s) and s != target]
    symbols = [target] + feat_syms
    cols = [f'close_{s}' for s in symbols]

    print(f"  Feature symbols: {symbols}")

    print("\n" + "-" * 50)
    print("BUILDING SEQUENCES")
    print("-" * 50)

    data = df[cols].values.astype(np.float32)
    scaler_df = df[cols].shift(cfg['n_ahead']).rolling(24 * 14).median()
    scaler_vals = scaler_df.values
    safe_scaler = np.where(scaler_vals < 1e-8, 1.0, scaler_vals)
    scaled = np.nan_to_num((data / safe_scaler).astype(np.float32), nan=1.0, posinf=1.0, neginf=1.0)

    warmup = 24 * 14 + cfg['n_ahead']
    seq_len = cfg['sequence_length']

    X, y = [], []
    for i in range(seq_len, len(scaled) - cfg['n_ahead'] + 1):
        X.append(scaled[i - seq_len:i])
        y.append(scaled[i, 0])

    X_arr = np.array(X)
    y_arr = np.array(y).reshape(-1, 1)
    print(f"  X shape: {X_arr.shape}")
    print(f"  y shape: {y_arr.shape}")
    print(f"  Dropped windows: {warmup}")

    print("\n" + "-" * 50)
    print("SPLITTING DATA (TRAIN / VAL / TEST)")
    print("-" * 50)
    split = cfg['train_test_split']
    gap = seq_len + cfg['n_ahead']
    usable = len(X_arr) - 2 * gap
    if usable <= 0:
        raise ValueError("Not enough samples for split.")
    train_n = int(usable * split)
    val_n = (usable - train_n) // 2
    test_n = usable - train_n - val_n
    val_start = train_n + gap
    test_start = val_start + val_n + gap
    X_tr = X_arr[:train_n]
    X_va = X_arr[val_start:val_start + val_n]
    X_te = X_arr[test_start:test_start + test_n]
    y_tr = y_arr[:train_n]
    y_va = y_arr[val_start:val_start + val_n]
    y_te = y_arr[test_start:test_start + test_n]
    print(f"  Training samples:   {len(X_tr)}")
    print(f"  Validation samples: {len(X_va)}")
    print(f"  Test samples:       {len(X_te)}")
    print(f"  Purge gap:          {gap}")

    print("\n" + "-" * 50)
    print("NORMALIZING DATA (MinMax)")
    print("-" * 50)
    xmin = X_tr.min(axis=(0, 1), keepdims=True)
    xmax = X_tr.max(axis=(0, 1), keepdims=True)
    print(f"  X_min range: [{xmin.min():.4f}, {xmax.max():.4f}]")
    print(f"  X_max range: [{xmin.min():.4f}, {xmax.max():.4f}]")
    xrange = jnp.where(xmax - xmin < 1e-8, 1e-8, xmax - xmin)
    X_tr = (X_tr - xmin) / xrange
    X_va = (X_va - xmin) / xrange
    X_te = (X_te - xmin) / xrange

    ymin = float(y_tr.min())
    ymax = float(y_tr.max())
    yrange = ymax - ymin if ymax - ymin > 1e-8 else 1.0
    y_tr = (y_tr - ymin) / yrange
    y_va = (y_va - ymin) / yrange
    y_te = (y_te - ymin) / yrange
    print(f"  y_min: {ymin:.4f}, y_max: {ymax:.4f}, y_range: {yrange:.4f}")
    print("  Normalization applied!")

    X_tr = jnp.array(X_tr)
    y_tr = jnp.array(y_tr)
    X_va = jnp.array(X_va)
    y_va = jnp.array(y_va)
    X_te = jnp.array(X_te)
    y_te = jnp.array(y_te)

    print(f"\n  Converted to JAX arrays:")
    print(f"    X_train: {X_tr.shape}")
    print(f"    y_train: {y_tr.shape}")
    print(f"    X_val:   {X_va.shape}")
    print(f"    y_val:   {y_va.shape}")
    print(f"    X_test:  {X_te.shape}")
    print(f"    y_test:  {y_te.shape}")
    print("-" * 50 + "\n")

    from tkan.tkan_init import init_tkan
    from tkan.tkan_forward import tkan_fwd
    from tkan.export import save_norm_params, save_config, to_onnx_regression

    hidden, sub = cfg['hidden_size'], cfg['sub_dim']
    input_dim = int(X_tr.shape[-1])
    epochs = cfg['epochs']
    lr = cfg['learning_rate']
    bs = cfg['batch_size']
    seed = cfg['seed']

    key = jax.random.key(seed)
    key, k = jax.random.split(key)
    params = init_tkan(input_dim, hidden, sub, k)
    print(f"\n=== TKAN ===")
    print(f"  Params: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    total_steps = max(1, len(range(0, len(X_tr), bs)) * epochs)
    warmup_steps = max(1, int(0.1 * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6, peak_value=lr,
        warmup_steps=warmup_steps, decay_steps=total_steps, end_value=1e-6,
    )
    opt = optax.adamw(schedule, weight_decay=1e-4)
    opt_st = opt.init(params)

    def apply_fn(params, x):
        h = tkan_fwd(params, x, hidden, sub)
        return jnp.dot(h, params['dense_w']) + params['dense_b']

    def mse_loss(params, x, y):
        return jnp.mean((apply_fn(params, x) - y) ** 2)

    @jax.jit
    def step(params, opt_st, x, y):
        loss, grad = jax.value_and_grad(mse_loss)(params, x, y)
        updates, opt_st = opt.update(grad, opt_st, params)
        params = optax.apply_updates(params, updates)
        return params, opt_st, loss

    train_losses, val_losses = [], []
    best_params = params
    best_val_loss = float('inf')
    best_epoch = 0
    start = time.time()
    num_batches = len(range(0, len(X_tr), bs))

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20, complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[green]Epochs", total=epochs)

        for ep in range(epochs):
            batch_task = progress.add_task("[green]Batches", total=num_batches)
            idx = jax.random.permutation(jax.random.key(seed + ep), len(X_tr))
            ep_loss = 0.0

            for i in range(0, len(X_tr), bs):
                b = idx[i:i + bs]
                params, opt_st, l = step(params, opt_st, X_tr[b], y_tr[b])
                ep_loss += float(l)
                progress.update(batch_task, advance=1)

            progress.remove_task(batch_task)
            train_loss = ep_loss / num_batches
            val_loss = float(mse_loss(params, X_va, y_va))

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            print(f"  epoch {ep + 1} | trainloss {train_loss:.4f} | valloss {val_loss:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_epoch = ep + 1

            progress.update(task, advance=1, description=f"[green]Epoch {ep + 1}/{epochs}")

    elapsed = time.time() - start

    test_preds_norm = apply_fn(best_params, X_te)
    test_preds = test_preds_norm * yrange + ymin
    test_mse = float(jnp.mean((test_preds - y_te * yrange + ymin) ** 2))
    test_rmse = float(np.sqrt(test_mse))
    ss_res = float(jnp.sum((y_te * yrange + ymin - test_preds) ** 2))
    ss_tot = float(jnp.sum((y_te * yrange + ymin - (y_te * yrange + ymin).mean()) ** 2))
    test_r2 = 1 - ss_res / ss_tot if ss_tot > 1e-8 else 0.0

    print(f"\n  Best epoch: {best_epoch}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test R2: {test_r2:.4f}")
    print(f"  Time: {elapsed:.1f}s")

    print("\n  Saving model outputs...")
    from tkan.export import save_norm_params_regression, save_config_regression, to_onnx_regression

    save_norm_params_regression(xmin.squeeze(), xmax.squeeze(), output_dir=str(model_dir))
    save_config_regression({**cfg, 'input_dim': input_dim, 'enabled_symbols': symbols}, output_dir=str(model_dir))

    with open(model_dir / 'config.yaml', 'w') as f:
        yaml.dump({**cfg, 'input_dim': input_dim}, f, default_flow_style=False)
    print(f"  Saved config.yaml to {model_dir}")

    notes = f"""# r.py Training Notes

## Best Epoch
- **Epoch**: {best_epoch} (of {cfg['epochs']} total)

## Training Metrics
- **Best Val Loss**: {best_val_loss:.6f}
- **Test RMSE**: {test_rmse:.4f}
- **Test R2**: {test_r2:.4f}
- **Total elapsed**: {elapsed:.1f} seconds

## Configuration
- **Sequence length**: {cfg['sequence_length']}
- **n_ahead**: {cfg['n_ahead']}
- **Hidden size**: {cfg['hidden_size']}
- **Sub dim**: {cfg['sub_dim']}
- **Batch size**: {cfg['batch_size']}
- **Learning rate**: {cfg['learning_rate']}
- **Seed**: {cfg['seed']}
- **Input dim**: {input_dim}
- **Symbols**: {symbols}

## Epoch History
"""
    for ep in range(len(train_losses)):
        notes += f"- Epoch {ep + 1}: train_loss={train_losses[ep]:.6f}, val_loss={val_losses[ep]:.6f}\n"

    with open(model_dir / 'notes.md', 'w') as f:
        f.write(notes)
    print(f"  Saved notes.md to {model_dir}")

    to_onnx_regression(best_params, sequence_length=seq_len, input_dim=input_dim, hidden=hidden, sub=sub, output_dir=str(model_dir))
    print(f"  Saved ONNX to {model_dir}")

    expert_path = Path('live.ex5')
    if expert_path.exists():
        latest_input = max(path.stat().st_mtime for path in [
            model_dir / 'model.onnx',
            model_dir / 'config.mqh',
            model_dir / 'norm_params.mqh',
        ])
        if expert_path.stat().st_mtime < latest_input:
            print("  live.ex5 is older than model. Recompile live.mq5 in MetaEditor.")

    expert_path = Path('live.ex5')
    r_mq5_path = Path('r.mq5')
    if expert_path.exists() or r_mq5_path.exists():
        latest_input = max(path.stat().st_mtime for path in [
            model_dir / 'model.onnx',
            model_dir / 'config.mqh',
            model_dir / 'norm_params.mqh',
        ])
        if r_mq5_path.exists() and r_mq5_path.stat().st_mtime < latest_input:
            content = r_mq5_path.read_text()
            ts = model_dir.name
            content = content.replace('#include "models/DDMM-HHMMSS/config.mqh"', f'#include "models/{ts}/config.mqh"')
            content = content.replace('#include "models/DDMM-HHMMSS/norm_params.mqh"', f'#include "models/{ts}/norm_params.mqh"')
            content = content.replace('#resource "\\\\Experts\\\\54\\\\models\\\\DDMM-HHMMSS\\\\model.onnx"', f'#resource "\\\\Experts\\\\54\\\\models\\\\{ts}\\\\model.onnx"')
            r_mq5_path.write_text(content)
            print("  r.mq5 updated to use model: ", ts)

    print("\n  Committing model to git...")
    subprocess.run(['git', 'add', '.'], check=True)
    subprocess.run(['git', 'commit', '-m', f'new model {model_dir.name}'], check=True)
    subprocess.run(['git', 'push'], check=True)
    print("  Model committed and pushed.")
    print("\nDone!")


if __name__ == '__main__':
    main()