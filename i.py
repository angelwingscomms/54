#!/usr/bin/env python3
"""
TKAN / GRU / LSTM trainer for forex/commodity OHLCV data.
Exports trained model to ONNX for use in MetaTrader 5 EAs.

Backend: JAX (default) or torch — no TensorFlow dependency.
ONNX export via Keras 3 native model.export(format="onnx").

Usage:
    python train.py                    # uses config.yaml
    python train.py --config my.yaml   # custom config
"""

# ── backend must be set before any keras import ──────────────────────────────
import os, sys, argparse, time, json

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="i.yaml")
args = parser.parse_args()

import yaml
with open(args.config) as f:
    cfg = yaml.safe_load(f)

BACKEND = cfg.get("backend", "jax")
if BACKEND == "tensorflow":
    sys.exit(
        "ERROR: backend 'tensorflow' is not supported by this script.\n"
        "Set backend to 'jax' (recommended) or 'torch' in your config."
    )
os.environ["KERAS_BACKEND"] = BACKEND

# ── stdlib / third-party ─────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, GRU

try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

from sklearn.metrics import r2_score
from tkan import TKAN

# ── config ────────────────────────────────────────────────────────────────────
DATA_PATH    = cfg["data_path"]
SYMBOL       = cfg["symbol"]
SEQ_LEN      = int(cfg.get("sequence_length", 54))
N_AHEAD      = int(cfg.get("n_ahead", 1))
N_EPOCHS     = int(cfg.get("epochs", 1000))
BATCH_SIZE   = int(cfg.get("batch_size", 128))
MODEL_TYPE   = cfg.get("model", "TKAN")
HIDDEN_UNITS = int(cfg.get("hidden_units", 100))
_feat_cfg    = cfg.get("features", "close")
FEATURES     = _feat_cfg if isinstance(_feat_cfg, list) else [_feat_cfg]
FEATURE_SYMS = [k for k, v in cfg.get("feature_symbols", {}).items() if v]
OUTPUT_PATH  = cfg.get("output_path", "model.onnx")
ROLLING_DAYS = int(cfg.get("rolling_days", 14))
OPSET        = int(cfg.get("onnx_opset", 13))

print(f"\n{'='*60}")
print(f"  Model      : {MODEL_TYPE}")
print(f"  Symbol     : {SYMBOL}")
print(f"  Features   : {FEATURES}")
print(f"  Feat syms  : {FEATURE_SYMS}")
print(f"  Seq len    : {SEQ_LEN}   N-ahead: {N_AHEAD}")
print(f"  Backend    : {BACKEND}")
print(f"{'='*60}\n")

# ── helpers ───────────────────────────────────────────────────────────────────
class MinMaxScaler:
    """Per-feature min-max scaler supporting 1-D, 2-D and 3-D arrays."""

    def __init__(self, feature_axis=None, minmax_range=(0.0, 1.0)):
        self.feature_axis = feature_axis
        self.minmax_range = minmax_range
        self.min_ = self.max_ = self.scale_ = None

    def fit(self, X):
        if X.ndim == 3 and self.feature_axis is not None:
            axis = tuple(i for i in range(X.ndim) if i != self.feature_axis)
            self.min_ = np.min(X, axis=axis)
            self.max_ = np.max(X, axis=axis)
        elif X.ndim == 2:
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        elif X.ndim == 1:
            self.min_ = float(np.min(X))
            self.max_ = float(np.max(X))
        else:
            raise ValueError("Data must be 1-D, 2-D or 3-D.")
        self.scale_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X):
        lo, hi = self.minmax_range
        return (X - self.min_) / self.scale_ * (hi - lo) + lo

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        lo, hi = self.minmax_range
        return (X_scaled - lo) / (hi - lo) * self.scale_ + self.min_

    def to_dict(self):
        return {
            "min_":        self.min_.tolist()   if hasattr(self.min_,   "tolist") else self.min_,
            "max_":        self.max_.tolist()   if hasattr(self.max_,   "tolist") else self.max_,
            "scale_":      self.scale_.tolist() if hasattr(self.scale_, "tolist") else self.scale_,
            "minmax_range": list(self.minmax_range),
        }


# ── data loading ──────────────────────────────────────────────────────────────
def _parse_datetime(s: str) -> str:
    """'2026-04-27 15-47'  →  '2026-04-27 15:47'"""
    s = str(s).strip()
    date_part, time_part = s.split(" ", 1)
    time_part = time_part.replace("-", ":")
    return f"{date_part} {time_part}"


def load_and_pivot(data_path, symbol, feature_syms, features):
    csv = data_path if data_path.endswith(".csv") else data_path + ".csv"
    raw = pd.read_csv(csv, encoding='utf-16')
    raw["datetime"] = pd.to_datetime(raw["datetime"].apply(_parse_datetime))
    raw = raw.set_index("datetime").sort_index()

    all_syms = [symbol] + feature_syms
    raw = raw[raw["symbol"].isin(all_syms)]

    parts = []
    for sym in all_syms:
        sub = raw[raw["symbol"] == sym][features].copy()
        sub.columns = [f"{sym}_{feat}" for feat in features]
        parts.append(sub)

    wide = pd.concat(parts, axis=1).sort_index().ffill().dropna()

    target_cols = [c for c in wide.columns if c.startswith(symbol + "_")]
    other_cols  = [c for c in wide.columns if not c.startswith(symbol + "_")]
    return wide[target_cols + other_cols], target_cols


wide, target_cols = load_and_pivot(DATA_PATH, SYMBOL, FEATURE_SYMS, FEATURES)
n_target_cols = len(target_cols)

print(f"Loaded data: {wide.shape}  ({wide.index[0]} → {wide.index[-1]})")
print(f"Columns: {list(wide.columns)}\n")


# ── dataset generation ────────────────────────────────────────────────────────
ROLLING_BARS = ROLLING_DAYS * 24   # hourly bars

def generate_sequences(df, seq_len, n_ahead):
    scaler_df = df.shift(n_ahead).rolling(ROLLING_BARS).median()
    tmp_df    = (df / scaler_df).iloc[ROLLING_BARS + n_ahead:].fillna(0.0)
    scaler_df = scaler_df.iloc[ROLLING_BARS + n_ahead:].fillna(0.0)

    X_list, y_list, ys_list = [], [], []
    for i in range(seq_len, len(tmp_df) - n_ahead + 1):
        X_list.append(tmp_df.iloc[i - seq_len : i].values)
        y_list.append(tmp_df.iloc[i : i + n_ahead, :n_target_cols].values)
        ys_list.append(scaler_df.iloc[i : i + n_ahead, :n_target_cols].values)

    X   = np.array(X_list,  dtype=np.float32)
    y   = np.array(y_list,  dtype=np.float32)
    y_s = np.array(ys_list, dtype=np.float32)

    split = int(len(X) * 0.8)
    Xtr_u, Xte_u = X[:split], X[split:]
    ytr_u, yte_u = y[:split], y[split:]

    Xs  = MinMaxScaler(feature_axis=2)
    Xtr = Xs.fit_transform(Xtr_u)
    Xte = Xs.transform(Xte_u)

    ys_sc = MinMaxScaler(feature_axis=2)
    ytr   = ys_sc.fit_transform(ytr_u).reshape(len(ytr_u), -1)
    yte   = ys_sc.transform(yte_u).reshape(len(yte_u), -1)

    return Xs, Xtr, Xte, ys_sc, ytr, yte, y_s[:split], y_s[split:], scaler_df


(X_scaler, X_train, X_test,
 y_scaler, y_train, y_test,
 ys_train, ys_test, norm_df) = generate_sequences(wide, SEQ_LEN, N_AHEAD)

n_out = y_train.shape[1]
print(f"X_train {X_train.shape}  y_train {y_train.shape}")
print(f"X_test  {X_test.shape}   y_test  {y_test.shape}\n")


# ── callbacks ─────────────────────────────────────────────────────────────────
def make_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-5, patience=10,
            mode="min", restore_best_weights=True, start_from_epoch=6,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.25, patience=5,
            mode="min", min_delta=1e-5, min_lr=2.5e-5, verbose=0,
        ),
        keras.callbacks.TerminateOnNaN(),
    ]


# ── build model ───────────────────────────────────────────────────────────────
keras.utils.set_random_seed(cfg.get("seed", 42))

input_shape = X_train.shape[1:]   # (seq_len, n_features)

if MODEL_TYPE == "TKAN":
    model = Sequential([
        Input(shape=input_shape),
        TKAN(HIDDEN_UNITS, return_sequences=True),
        TKAN(HIDDEN_UNITS,
             sub_kan_output_dim=20, sub_kan_input_dim=20,
             return_sequences=False),
        Dense(n_out, activation="linear"),
    ], name="TKAN")

elif MODEL_TYPE == "GRU":
    model = Sequential([
        Input(shape=input_shape),
        GRU(HIDDEN_UNITS, return_sequences=True),
        GRU(HIDDEN_UNITS, return_sequences=False),
        Dense(n_out, activation="linear"),
    ], name="GRU")

elif MODEL_TYPE == "LSTM":
    model = Sequential([
        Input(shape=input_shape),
        LSTM(HIDDEN_UNITS, return_sequences=True),
        LSTM(HIDDEN_UNITS, return_sequences=False),
        Dense(n_out, activation="linear"),
    ], name="LSTM")

else:
    sys.exit(f"Unknown model type '{MODEL_TYPE}'. Choose TKAN | GRU | LSTM.")

model.compile(
    optimizer=keras.optimizers.Adam(cfg.get("lr", 0.001)),
    loss="mean_squared_error",
)
model.summary()

# ── train ─────────────────────────────────────────────────────────────────────
t0 = time.time()
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=N_EPOCHS,
    validation_split=0.2,
    callbacks=make_callbacks(),
    shuffle=True,
    verbose=1,
)
elapsed = time.time() - t0
print(f"\nTraining finished in {elapsed:.1f}s")

# ── evaluate ──────────────────────────────────────────────────────────────────
preds = model.predict(X_test, verbose=0)
r2    = r2_score(y_test, preds)
rmse  = root_mean_squared_error(y_test, preds)
print(f"Test  R²={r2:.4f}   RMSE={rmse:.6f}\n")

# ── export ONNX ───────────────────────────────────────────────────────────────
# Keras 3 native export — works with JAX and torch backends, no tf2onnx needed.
# The opset_version kwarg was added in Keras 3.x; older builds ignore it and
# default to opset 18, which MT5 may not handle — upgrade keras if needed.
print(f"Exporting ONNX → {OUTPUT_PATH}  (opset {OPSET})")
model.export(OUTPUT_PATH, format="onnx", opset_version=OPSET)
print(f"Saved: {OUTPUT_PATH}")

# ── quick ONNX sanity check ───────────────────────────────────────────────────
onnx_input_name = "input"
try:
    import onnxruntime as ort
    sess = ort.InferenceSession(OUTPUT_PATH, providers=["CPUExecutionProvider"])
    onnx_input_name = sess.get_inputs()[0].name
    sample    = X_test[:4].astype(np.float32)
    ort_out   = sess.run(None, {onnx_input_name: sample})[0]
    keras_out = model.predict(sample, verbose=0)
    max_diff  = float(np.max(np.abs(ort_out - keras_out)))
    status    = "✓" if max_diff < 1e-4 else "⚠ check model"
    print(f"ONNX sanity check  max|Δ| = {max_diff:.2e}  {status}")
except ImportError:
    print("onnxruntime not installed – skipping sanity check")

# ── save scaler + metadata for the EA ────────────────────────────────────────
meta = {
    "symbol":          SYMBOL,
    "feature_symbols": FEATURE_SYMS,
    "features":        FEATURES,
    "all_columns":     list(wide.columns),
    "target_columns":  target_cols,
    "sequence_length": SEQ_LEN,
    "n_ahead":         N_AHEAD,
    "rolling_bars":    ROLLING_BARS,
    "model_type":      MODEL_TYPE,
    "onnx_input_name": onnx_input_name,
    "X_scaler":        X_scaler.to_dict(),
    "y_scaler":        y_scaler.to_dict(),
    "metrics": {
        "r2":            round(r2, 6),
        "rmse":          round(rmse, 8),
        "train_seconds": round(elapsed, 1),
    },
}

meta_path = OUTPUT_PATH.replace(".onnx", "_meta.json")
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved: {meta_path}")
print("\nDone ✓")