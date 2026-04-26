import os
os.environ['JAX_CPU_COLLECTIVE_IMPL_HEADER_ONLY'] = '1'

from datetime import datetime
from pathlib import Path

import jax
import jax.numpy as jnp
import pandas as pd
import yaml
from tkan import (
    load_config, load_csv, compute_atr, build_samples,
    normalize, save_norm_params, save_config, to_onnx_model, train, eval_loss, tkan_apply,
    build_feature_frame, select_symbol_ohlc,
)

jax.default_backend = 'cpu'


def main():
    print("\n" + "#"*60)
    print("# TKAN TRAINING PIPELINE STARTING")
    print("#"*60 + "\n")

    cfg = load_config()
    model_dir = Path(f"/models/{datetime.now().strftime('%m%d-%H%M%S')}")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Model output directory: {model_dir}")
    seq_len = cfg['sequence_length']

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
    print(f"    - target_symbol: {cfg['symbol']}")
    print(f"    - feature_symbols: {cfg['enabled_symbols']}")

    print(f"\n  Loading CSV file...")
    df = load_csv(f'./data/{cfg["data_path"]}')
    print(f"  CSV loaded: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")

    features = build_feature_frame(df, cfg)
    target = select_symbol_ohlc(df, cfg['symbol'])
    merged = pd.concat([features, target.add_prefix('target_')], axis=1).dropna()
    features = merged[features.columns].copy()
    target = merged[[f'target_{field}' for field in ('open', 'high', 'low', 'close')]].copy()
    target.columns = ['open', 'high', 'low', 'close']
    print(f"  Feature data: {features.shape}")
    print(f"  Target OHLC: {target.shape}")

    print(f"\n  Computing ATR (period={cfg['atr_period']})...")
    atr = compute_atr(target, cfg['atr_period'])
    print(f"  ATR computed, first valid ATR at index: {atr.first_valid_index()}")

    print(f"\n  Building training samples (sequence_length={seq_len}, horizon={cfg['n_ahead']})...")
    X_arr, y_arr = build_samples(
        features, target, atr,
        sequence_length=seq_len,
        horizon=cfg['n_ahead'],
        tp_pct=cfg['threshold_pct'],
        tolerance=cfg['stop_loss_pct'],
        target_type=cfg['target_type'],
        atr_multiplier=cfg['atr_multiplier'],
        tp_multiplier=cfg['tp_multiplier']
    )

    total_candidates = max(0, len(features) - seq_len - cfg['n_ahead'] + 1)
    buy_count = int(y_arr.sum())
    sell_count = len(y_arr) - buy_count
    dropped_count = total_candidates - len(X_arr)
    if len(X_arr) == 0:
        raise ValueError("No clean labeled samples were produced. Too many windows were dropped as ambiguous or unresolved.")
    print(f"\n  Done! Generated {len(X_arr)} clean samples")
    print(f"    - X shape: {X_arr.shape}")
    print(f"    - y shape: {y_arr.shape}")
    print(f"    - Buy labels:  {buy_count} ({100*buy_count/len(y_arr):.1f}%)")
    print(f"    - Sell labels: {sell_count} ({100*sell_count/len(y_arr):.1f}%)")
    print(f"    - Dropped windows: {dropped_count}")
    print("="*50 + "\n")

    print("\n" + "-"*50)
    print("SPLITTING DATA (TRAIN / VAL / TEST WITH PURGE GAP)")
    print("-"*50)
    split = cfg['train_test_split']
    gap = seq_len + cfg['n_ahead']
    if not 0 < split < 1:
        raise ValueError("train_test_split must be between 0 and 1.")
    usable = len(X_arr) - 2 * gap
    if usable <= 0:
        raise ValueError("Not enough samples for a purged train/val/test split. Add more data or reduce sequence_length/horizon.")
    train_n = int(usable * split)
    val_n = (usable - train_n) // 2
    test_n = usable - train_n - val_n
    val_start = train_n + gap
    test_start = val_start + val_n + gap
    X_tr, X_va, X_te = X_arr[:train_n], X_arr[val_start:val_start + val_n], X_arr[test_start:test_start + test_n]
    y_tr, y_va, y_te = y_arr[:train_n], y_arr[val_start:val_start + val_n], y_arr[test_start:test_start + test_n]
    if min(len(X_tr), len(X_va), len(X_te)) == 0:
        raise ValueError("Not enough samples for a purged train/val/test split. Add more data or reduce sequence_length/horizon.")
    print(f"  Training samples:   {len(X_tr)}")
    print(f"  Validation samples:{len(X_va)}")
    print(f"  Test samples:      {len(X_te)}")
    print(f"  Purge gap:         {gap}")

    print("\n" + "-"*50)
    print("NORMALIZING DATA")
    print("-"*50)
    xmin, xmax = X_tr.min(axis=(0, 1), keepdims=True), X_tr.max(axis=(0, 1), keepdims=True)
    print(f"  X_min range: [{xmin.min():.4f}, {xmin.max():.4f}]")
    print(f"  X_max range: [{xmax.min():.4f}, {xmax.max():.4f}]")
    X_tr, X_va, X_te, y_tr, y_va, y_te = normalize(xmin, xmax, X_tr, X_va, X_te, y_tr, y_va, y_te)
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
    print("-"*50 + "\n")

    hidden, sub = cfg['hidden_size'], cfg['sub_dim']
    input_dim = X_tr.shape[-1]

    print("\n=== TKAN ===")
    params, train_losses, val_losses, train_accs, val_accs, elapsed = train(
        X_tr,
        y_tr,
        X_va,
        y_va,
        X_te,
        y_te,
        input_dim,
        hidden,
        sub,
        epochs=cfg['epochs'],
        lr=cfg['learning_rate'],
        batch_size=cfg['batch_size'],
        seed=cfg['seed'],
    )

    cfg['input_dim'] = int(input_dim)
    save_norm_params(xmin, xmax, output_dir=str(model_dir))
    save_config(cfg, output_dir=str(model_dir))

    with open(model_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Saved config.yaml to {model_dir}")

    print("\nExporting model to ONNX...")
    to_onnx_model(params, sequence_length=seq_len, input_dim=input_dim, hidden=hidden, sub=sub, output_dir=str(model_dir))
    model_path = model_dir / 'model.onnx'
    config_path = model_dir / 'config.mqh'
    norm_path = model_dir / 'norm_params.mqh'
    expert_path = Path('live.ex5')
    latest_input = max(path.stat().st_mtime for path in (model_path, config_path, norm_path))
    if not expert_path.exists() or expert_path.stat().st_mtime < latest_input:
        print("live.ex5 is older than model.onnx/config.mqh/norm_params.mqh. Recompile live.mq5 in MetaEditor before running the tester.")


if __name__ == '__main__':
    main()
