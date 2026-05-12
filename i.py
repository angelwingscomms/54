#!/usr/bin/env python3
"""
TKAN / GRU / LSTM trainer for forex/commodity OHLCV data.
Exports trained model to ONNX for use in MetaTrader 5 EAs.
Applies an MQL5 Compatibility Patch to bypass ERR_ONNX_INVALID_SHAPE (5805).
"""

import os, sys, argparse, time, json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="i.yaml")
parser.add_argument("-n", "--name", default=None, help="model folder name (default: ddmm-hhmmss)")
args = parser.parse_args()

MODEL_FOLDER_NAME = args.name
if MODEL_FOLDER_NAME is None:
    now = datetime.now()
    MODEL_FOLDER_NAME = f"{now.day:02d}{now.month:02d}-{now.hour:02d}{now.minute:02d}{now.second:02d}"

with open(args.config) as f:
    cfg = yaml.safe_load(f)

BACKEND = cfg.get("backend", "jax")
os.environ["KERAS_BACKEND"] = BACKEND

import keras
from keras.layers import LSTM, Dense, Input
from keras.models import Sequential
from tkan import TKAN

def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

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
FEATURE_SYMS =[k for k, v in cfg.get("feature_symbols", {}).items() if v]
ROLLING_DAYS = int(cfg.get("rolling_days", 14))
OPSET        = int(cfg.get("onnx_opset", 18))
TARGET_MODE  = str(cfg.get("target_mode", "price")).strip().lower()

MODEL_OUT_DIR = os.path.join("./models", MODEL_FOLDER_NAME)
os.makedirs(MODEL_OUT_DIR, exist_ok=True)

OUTPUT_PATH = os.path.join(MODEL_OUT_DIR, "model.onnx")

MQL5_FEATURES = {"open", "high", "low", "close", "tick_volume", "volume", "vol", "real_volume", "spread"}

def clean_feature_name(feature):
    if feature is None:
        return "close"
    feature = str(feature).strip().lower()
    return "close" if feature in ("", "none", "null") else feature

def unique_nonempty(values):
    out, seen = [], set()
    for value in values:
        text = str(value).strip()
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out

SYMBOL = str(SYMBOL).strip()
FEATURES = [clean_feature_name(f) for f in FEATURES]
FEATURE_SYMS = [s for s in unique_nonempty(FEATURE_SYMS) if s != SYMBOL]
ALL_SYMS = [SYMBOL] + FEATURE_SYMS

unsupported = sorted({f for f in FEATURES if f not in MQL5_FEATURES})
if unsupported:
    raise ValueError(
        "features must be reproducible by i.mq5 at runtime; unsupported: "
        + ", ".join(unsupported)
    )

if TARGET_MODE not in ("price", "range_close"):
    raise ValueError("target_mode must be 'price' or 'range_close'")

if TARGET_MODE == "range_close":
    if N_AHEAD != 1:
        raise ValueError("target_mode=range_close currently requires n_ahead: 1")
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(FEATURES))
    if missing:
        raise ValueError("target_mode=range_close requires input features: " + ", ".join(missing))

# ── helpers ───────────────────────────────────────────────────────────────────
class MinMaxScaler:
    def __init__(self, feature_axis=None, minmax_range=(0.0, 1.0)):
        self.feature_axis = feature_axis
        self.minmax_range = minmax_range

    def fit(self, X):
        if X.ndim == 3 and self.feature_axis is not None:
            axis = tuple(i for i in range(X.ndim) if i != self.feature_axis)
            self.min_ = np.min(X, axis=axis)
            self.max_ = np.max(X, axis=axis)
        elif X.ndim == 2:
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        else:
            self.min_ = float(np.min(X))
            self.max_ = float(np.max(X))
        self.scale_ = np.where(self.max_ - self.min_ == 0, 1.0, self.max_ - self.min_)
        return self

    def transform(self, X):
        lo, hi = self.minmax_range
        return (X - self.min_) / self.scale_ * (hi - lo) + lo

    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ── data loading & dataset generation ─────────────────────────────────────────
def _parse_datetime(s: str) -> str:
    s = str(s).strip()
    date_part, time_part = s.split(" ", 1)
    return f"{date_part} {time_part.replace('-', ':')}"

def load_and_pivot(data_path, symbol, all_syms, features):
    csv = data_path if data_path.endswith(".csv") else data_path + ".csv"
    raw = pd.read_csv(csv, encoding='utf-16')
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    raw["datetime"] = pd.to_datetime(raw["datetime"].apply(_parse_datetime))
    raw = raw.set_index("datetime").sort_index()
    raw = raw[raw["symbol"].isin(all_syms)]
    missing = [f for f in features if f not in raw.columns]
    if missing:
        raise ValueError("CSV is missing configured feature columns: " + ", ".join(missing))

    parts = []
    feature_pairs = []
    for sym in all_syms:
        sub = raw[raw["symbol"] == sym][features].copy()
        sub.columns =[f"{sym}_{feat}" for feat in features]
        parts.append(sub)
        feature_pairs.extend((sym, feat) for feat in features)

    wide = pd.concat(parts, axis=1, sort=False).sort_index().ffill().dropna()
    target_cols =[c for c in wide.columns if c.startswith(symbol + "_")]
    other_cols  =[c for c in wide.columns if not c.startswith(symbol + "_")]
    ordered_cols = target_cols + other_cols
    pair_by_col = {f"{sym}_{feat}": (sym, feat) for sym, feat in feature_pairs}
    ordered_pairs = [pair_by_col[col] for col in ordered_cols]
    return wide[ordered_cols], target_cols, ordered_pairs

wide, target_cols, feature_pairs = load_and_pivot(DATA_PATH, SYMBOL, ALL_SYMS, FEATURES)
n_target_cols = len(target_cols)

ROLLING_BARS = ROLLING_DAYS * 24
def generate_sequences(df, seq_len, n_ahead):
    scaler_df = df.shift(n_ahead).rolling(ROLLING_BARS).median()
    tmp_df    = (df / scaler_df).iloc[ROLLING_BARS + n_ahead:].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    scaler_df = scaler_df.iloc[ROLLING_BARS + n_ahead:].fillna(0.0)

    X_list, y_list = [],[]
    for i in range(seq_len, len(tmp_df) - n_ahead + 1):
        X_list.append(tmp_df.iloc[i - seq_len : i].values)
        if TARGET_MODE == "range_close":
            row = df.loc[tmp_df.index[i]]
            scale = scaler_df.loc[tmp_df.index[i], f"{SYMBOL}_close"]
            if not np.isfinite(scale) or scale <= 0:
                scale = row[f"{SYMBOL}_close"]
            if not np.isfinite(scale) or scale <= 0:
                scale = 1.0

            bar_open = row[f"{SYMBOL}_open"]
            up_range = max(row[f"{SYMBOL}_high"] - bar_open, 0.0)
            down_range = max(bar_open - row[f"{SYMBOL}_low"], 0.0)
            y_list.append([
                row[f"{SYMBOL}_close"] / scale,
                up_range / scale,
                down_range / scale,
            ])
        else:
            y_list.append(tmp_df.iloc[i : i + n_ahead, :n_target_cols].values)

    X, y = np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)
    split = int(len(X) * 0.8)
    
    Xs = MinMaxScaler(feature_axis=2)
    Xtr = Xs.fit_transform(X[:split])
    Xte = Xs.transform(X[split:])

    ys_sc = MinMaxScaler(feature_axis=2)
    ytr = ys_sc.fit_transform(y[:split]).reshape(len(y[:split]), -1)
    yte = ys_sc.transform(y[split:]).reshape(len(y[split:]), -1)

    return Xs, Xtr, Xte, ys_sc, ytr, yte

X_scaler, X_train, X_test, y_scaler, y_train, y_test = generate_sequences(wide, SEQ_LEN, N_AHEAD)
n_out = y_train.shape[1]
TARGET_NAMES = ["close", "up_range", "down_range"] if TARGET_MODE == "range_close" else [
    col.split("_", 1)[1] for col in target_cols for _ in range(N_AHEAD)
]

# ── model & training ──────────────────────────────────────────────────────────
keras.utils.set_random_seed(cfg.get("seed", 42))
input_shape = X_train.shape[1:]

model = Sequential([
    Input(shape=input_shape),
    TKAN(HIDDEN_UNITS, return_sequences=True) if MODEL_TYPE=="TKAN" else LSTM(HIDDEN_UNITS, return_sequences=True),
    TKAN(HIDDEN_UNITS, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False) if MODEL_TYPE=="TKAN" else LSTM(HIDDEN_UNITS, return_sequences=False),
    Dense(n_out, activation="linear"),
])
model.compile(optimizer=keras.optimizers.Adam(cfg.get("lr", 0.001)), loss="mse")

t0 = time.time()
history = model.fit(
    X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=1
)
elapsed = time.time() - t0

y_pred = np.asarray(model.predict(X_test, verbose=0))
test_loss = float(model.evaluate(X_test, y_test, verbose=0))
test_rmse = root_mean_squared_error(y_test, y_pred)
test_r2 = float(r2_score(y_test, y_pred, multioutput="uniform_average"))
best_val_loss = min(history.history.get("val_loss", [float("nan")]))

print("\nEvaluation:")
print(f"  Best validation loss (MSE): {best_val_loss:.8f}")
print(f"  Test loss (MSE):           {test_loss:.8f}")
print(f"  Test RMSE:                 {test_rmse:.8f}")
print(f"  Test R2:                   {test_r2:.6f}")
print(f"  Training time:             {elapsed:.1f}s")

# ── export base ONNX ──────────────────────────────────────────────────────────
model.export(OUTPUT_PATH, format="onnx", opset_version=OPSET)

# ── MQL5 ONNX COMPATIBILITY PATCH ─────────────────────────────────────────────
import onnx
from onnx import helper, TensorProto

def make_mql5_compatible(onnx_path, seq_len, flat_features_in, flat_features_out):
    m_onnx = onnx.load(onnx_path)
    graph = m_onnx.graph
    
    if not graph.input: return
        
    # --- 1. FIX INPUT ---
    model_input = graph.input[0]
    input_name = model_input.name
    
    flat_size = seq_len * flat_features_in
    
    del model_input.type.tensor_type.shape.dim[:]
    d0 = model_input.type.tensor_type.shape.dim.add()
    d0.dim_value = 1  # Force Batch = 1
    
    d1 = model_input.type.tensor_type.shape.dim.add()
    d1.dim_value = flat_size
    
    # Create the internal Reshape nodes so the model still sees 3D internally
    reshaped_name = f"{input_name}_reshaped"
    for node in graph.node:
        for i, node_input in enumerate(node.input):
            if node_input == input_name:
                node.input[i] = reshaped_name
                
    shape_const_name = f"{input_name}_shape_const"
    shape_tensor = helper.make_tensor(
        name=f"{shape_const_name}_val",
        data_type=TensorProto.INT64,
        dims=[3],
        vals=[1, seq_len, flat_features_in] # Re-apply exact feature span
    )
    const_node = helper.make_node('Constant', inputs=[], outputs=[shape_const_name], value=shape_tensor)
    reshape_node = helper.make_node('Reshape', inputs=[input_name, shape_const_name], outputs=[reshaped_name])
    
    graph.node.insert(0, reshape_node)
    graph.node.insert(0, const_node)

    # --- 2. FIX OUTPUT ---
    if graph.output:
        model_output = graph.output[0]
        # Force Output matches n_out exactly
        del model_output.type.tensor_type.shape.dim[:]
        out_d0 = model_output.type.tensor_type.shape.dim.add()
        out_d0.dim_value = 1 # Force Batch = 1
        
        out_d1 = model_output.type.tensor_type.shape.dim.add()
        out_d1.dim_value = flat_features_out

    # Save and check
    onnx.checker.check_model(m_onnx)
    onnx.save(m_onnx, onnx_path)
    print(f"\n[ONNX Patch] MQL5-compatible ONNX saved.")
    print(f"            Input: [1, {flat_size}] -> [1, {seq_len}, {flat_features_in}]")
    print(f"            Output: [1, {flat_features_out}]")

make_mql5_compatible(OUTPUT_PATH, SEQ_LEN, input_shape[1], n_out)

# ── save .mqh arrays ──────────────────────────────────────────────────────────
mqh_path = os.path.join(MODEL_OUT_DIR, "model_meta.mqh")

def to_list(x):
    if hasattr(x, "tolist"): return x.tolist()
    return list(x) if isinstance(x, (list, tuple)) else[x]

x_min_list, x_scale_list = to_list(X_scaler.min_), to_list(X_scaler.scale_)
y_min_list, y_scale_list = to_list(y_scaler.min_), to_list(y_scaler.scale_)

flat_features_in = input_shape[1]
if len(feature_pairs) != flat_features_in:
    raise RuntimeError(
        f"metadata feature count mismatch: {len(feature_pairs)} names for {flat_features_in} model inputs"
    )

with open(mqh_path, "w") as f:
    f.write(f"const int    MODEL_SEQ_LEN      = {SEQ_LEN};\n")
    f.write(f"const int    MODEL_N_AHEAD      = {N_AHEAD};\n")
    f.write(f"const int    MODEL_ROLLING_DAYS = {ROLLING_DAYS};\n")
    f.write(f"const int    MODEL_N_FEATURES   = {flat_features_in};\n")
    f.write(f"const int    MODEL_N_OUT        = {n_out};\n\n")
    f.write(f"const string MODEL_PRIMARY_SYMBOL = {json.dumps(SYMBOL)};\n")
    f.write(f"const string MODEL_TARGET_MODE    = {json.dumps(TARGET_MODE)};\n")
    target_array = [json.dumps(name) for name in TARGET_NAMES]
    f.write(f"const string MODEL_TARGET_NAMES[{len(TARGET_NAMES)}] = {{{', '.join(target_array)}}};\n\n")

    sym_array = [json.dumps(sym) for sym, _ in feature_pairs]
    feat_array = [json.dumps(feat) for _, feat in feature_pairs]

    f.write(f"const string MODEL_SYMBOLS[{flat_features_in}] = {{{', '.join(sym_array)}}};\n")
    f.write(f"const string MODEL_FEATURE_NAMES[{flat_features_in}] = {{{', '.join(feat_array)}}};\n\n")

    f.write(f"const double MODEL_X_MIN[{len(x_min_list)}]   = {{{', '.join([f'{v:.8g}' for v in x_min_list])}}};\n")
    f.write(f"const double MODEL_X_SCALE[{len(x_scale_list)}] = {{{', '.join([f'{v:.8g}' for v in x_scale_list])}}};\n\n")

    f.write(f"const double MODEL_Y_MIN[{len(y_min_list)}]   = {{{', '.join([f'{v:.8g}' for v in y_min_list])}}};\n")
    f.write(f"const double MODEL_Y_SCALE[{len(y_scale_list)}] = {{{', '.join([f'{v:.8g}' for v in y_scale_list])}}};\n")

print(f"Saved: {mqh_path}\n")

live_mq5 = Path("i.mq5")
if live_mq5.exists():
    content = live_mq5.read_text()
    content = content.replace('#include "model_meta.mqh"', f'#include "models/{MODEL_FOLDER_NAME}/model_meta.mqh"')
    content = content.replace('#resource "model.onnx"', f'#resource "\\\\Experts\\\\54\\\\models\\\\{MODEL_FOLDER_NAME}\\\\model.onnx"')
    live_mq5.write_text(content)
    print(f"Updated i.mq5 to use model: {MODEL_FOLDER_NAME}")

print("Done ✓")
