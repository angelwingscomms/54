import os
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

import yaml
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tkan import (
    load_config, load_csv, compute_atr, build_samples,
    normalize, save_norm_params, save_config, to_onnx_model, train
)

jax.default_backend = 'cpu'


def main():
    print("\n" + "#"*60)
    print("# TKAN TRAINING PIPELINE STARTING")
    print("#"*60 + "\n")

    cfg = load_config()

    print("\n" + "="*50)
    print("LOADING DATA")
    print("="*50)
    print(f"  Path: {cfg['data_path']}")
    print(f"  Parameters:")
    print(f"    - tp_pct: {cfg['threshold_pct']}")
    print(f"    - tolerance: {cfg['stop_loss_pct']}")
    print(f"    - horizon: {cfg['n_ahead']}")
    print(f"    - target_type: {cfg['target_type']}")
    print(f"    - atr_multiplier: {cfg['atr_multiplier']}")
    print(f"    - tp_multiplier: {cfg['tp_multiplier']}")
    print(f"    - atr_period: {cfg['atr_period']}")

    print(f"\n  Loading CSV file...")
    df = load_csv(cfg['data_path'])
    print(f"  CSV loaded: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    btc = df[['open', 'high', 'low', 'close']].copy()
    print(f"  Extracted OHLC data: {btc.shape}")

    print(f"\n  Computing ATR (period={cfg['atr_period']})...")
    atr = compute_atr(btc, cfg['atr_period'])
    print(f"  ATR computed, first valid ATR at index: {atr.first_valid_index()}")

    print(f"\n  Building training samples (sequence_length=45, horizon={cfg['n_ahead']})...")
    X_arr, y_arr = build_samples(
        btc, atr,
        horizon=cfg['n_ahead'],
        tp_pct=cfg['threshold_pct'],
        tolerance=cfg['stop_loss_pct'],
        target_type=cfg['target_type'],
        atr_multiplier=cfg['atr_multiplier'],
        tp_multiplier=cfg['tp_multiplier']
    )

    pos_count = int(y_arr.sum())
    neg_count = len(y_arr) - pos_count
    print(f"\n  Done! Generated {len(X_arr)} samples")
    print(f"    - X shape: {X_arr.shape}")
    print(f"    - y shape: {y_arr.shape}")
    print(f"    - Positive (TP hit): {pos_count} ({100*pos_count/len(y_arr):.1f}%)")
    print(f"    - Negative (SL hit): {neg_count} ({100*neg_count/len(y_arr):.1f}%)")
    print("="*50 + "\n")

    print("\n" + "-"*50)
    print("SPLITTING DATA (80% train / 20% test)")
    print("-"*50)
    sep = int(len(X_arr) * 0.8)
    X_tr, X_te = X_arr[:sep], X_arr[sep:]
    y_tr, y_te = y_arr[:sep], y_arr[sep:]
    print(f"  Training samples:   {len(X_tr)}")
    print(f"  Test samples:      {len(X_te)}")

    print("\n" + "-"*50)
    print("NORMALIZING DATA")
    print("-"*50)
    xmin, xmax = X_tr.min(axis=(0, 1), keepdims=True), X_tr.max(axis=(0, 1), keepdims=True)
    print(f"  X_min range: [{xmin.min():.4f}, {xmin.max():.4f}]")
    print(f"  X_max range: [{xmax.min():.4f}, {xmax.max():.4f}]")
    X_tr, X_te, y_tr, y_te = normalize(xmin, xmax, X_tr, X_te, y_tr, y_te)
    print("  Normalization applied!")

    X_tr = jnp.array(X_tr)
    y_tr = jnp.array(y_tr)
    X_te = jnp.array(X_te)
    y_te = jnp.array(y_te)

    print(f"\n  Converted to JAX arrays:")
    print(f"    X_train: {X_tr.shape}")
    print(f"    y_train: {y_tr.shape}")
    print(f"    X_test:  {X_te.shape}")
    print(f"    y_test:  {y_te.shape}")
    print("-"*50 + "\n")

    hidden, sub = 100, 20
    input_dim = X_tr.shape[-1]

    print("\n=== TKAN ===")
    params, train_losses, val_losses, acc, elapsed = train(
        X_tr, y_tr, X_te, y_te, input_dim, hidden, sub, epochs=cfg['epochs'], lr=cfg['learning_rate']
    )

    print("\n" + "="*48)
    print("SUMMARY")
    print("="*48)
    print(f"{'Epoch':>6} | {'Train':>8} | {'Val':>8}")
    print("-"*48)
    for i, (tl, vl) in enumerate(zip(train_losses, val_losses)):
        print(f"{i+1:>6} | {tl:>8.4f} | {vl:>8.4f}")
    print("="*48)
    print(f"TKAN time: {elapsed:.1f}s  Final val_acc: {acc:.4f}")

    save_norm_params(xmin, xmax)
    save_config(cfg)

    print("\nExporting model to ONNX...")
    to_onnx_model(params)
    print(f"Model saved to: model.onnx")


if __name__ == '__main__':
    main()