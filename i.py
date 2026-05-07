#!/usr/bin/env python3
"""
TKAN / GRU / LSTM trainer for forex/commodity OHLCV data.
Exports trained model to ONNX for use in MetaTrader 5 EAs.
Applies an MQL5 Compatibility Patch to bypass ERR_ONNX_INVALID_SHAPE (5805).
"""

import os, sys, argparse, time, json
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="i.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

BACKEND = cfg.get("backend", "jax")
os.environ["KERAS_BACKEND"] = BACKEND

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input, GRU
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
def root_mean_squared_error(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))
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
FEATURE_SYMS =[k for k, v in cfg.get("feature_symbols", {}).items() if v]
OUTPUT_PATH  = cfg.get("output_path", "model.onnx")
ROLLING_DAYS = int(cfg.get("rolling_days", 14))
OPSET        = int(cfg.get("onnx_opset", 18))

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
        sub.columns =[f"{sym}_{feat}" for feat in features]
        parts.append(sub)
    wide = pd.concat(parts, axis=1).sort_index().ffill().dropna()
    target_cols =[c for c in wide.columns if c.startswith(symbol + "_")]
    other_cols  =[c for c in wide.columns if not c.startswith(symbol + "_")]
    return wide[target_cols + other_cols], target_cols

wide, target_cols = load_and_pivot(DATA_PATH, SYMBOL, FEATURE_SYMS, FEATURES)
n_target_cols = len(target_cols)

ROLLING_BARS = ROLLING_DAYS * 24
def generate_sequences(df, seq_len, n_ahead):
    scaler_df = df.shift(n_ahead).rolling(ROLLING_BARS).median()
    tmp_df    = (df / scaler_df).iloc[ROLLING_BARS + n_ahead:].fillna(0.0)
    scaler_df = scaler_df.iloc[ROLLING_BARS + n_ahead:].fillna(0.0)

    X_list, y_list = [],[]
    for i in range(seq_len, len(tmp_df) - n_ahead + 1):
        X_list.append(tmp_df.iloc[i - seq_len : i].values)
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
model.fit(
    X_train, y_train, batch_size=BATCH_SIZE, epochs=N_EPOCHS, validation_split=0.2,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)],
    verbose=1
)
elapsed = time.time() - t0

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

# ── After model.export(...) in your script ──────────────────────────────────

# ── After model.export(...) in your script ──────────────────────────────────
# Clean feature strings to prevent empty/None values in MQL5
FEATURES = [str(f).strip().lower() for f in FEATURES]
FEATURES =[f if f and f != "none" else "close" for f in FEATURES]

all_syms = [SYMBOL] + FEATURE_SYMS
make_mql5_compatible(OUTPUT_PATH, SEQ_LEN, input_shape[1], n_out)

# ── save .mqh arrays ──────────────────────────────────────────────────────────
mqh_path = OUTPUT_PATH.replace(".onnx", "_meta.mqh")

def to_list(x):
    if hasattr(x, "tolist"): return x.tolist()
    return list(x) if isinstance(x, (list, tuple)) else[x]

x_min_list, x_scale_list = to_list(X_scaler.min_), to_list(X_scaler.scale_)
y_min_list, y_scale_list = to_list(y_scaler.min_), to_list(y_scaler.scale_)

flat_features_in = input_shape[1]

with open(mqh_path, "w") as f:
    f.write(f"const int    MODEL_SEQ_LEN      = {SEQ_LEN};\n")
    f.write(f"const int    MODEL_N_AHEAD      = {N_AHEAD};\n")
    f.write(f"const int    MODEL_ROLLING_DAYS = {ROLLING_DAYS};\n")
    f.write(f"const int    MODEL_N_FEATURES   = {flat_features_in};\n")
    f.write(f"const int    MODEL_N_OUT        = {n_out};\n\n")

    sym_array = []
    feat_array =[]
    for sym in all_syms:
        for feat in FEATURES:
            sym_array.append(f'"{sym}"')
            feat_array.append(f'"{feat}"')

    f.write(f"const string MODEL_SYMBOLS[{flat_features_in}] = {{{', '.join(sym_array)}}};\n")
    f.write(f"const string MODEL_FEATURE_NAMES[{flat_features_in}] = {{{', '.join(feat_array)}}};\n\n")

    f.write(f"const double MODEL_X_MIN[{len(x_min_list)}]   = {{{', '.join([f'{v:.8g}' for v in x_min_list])}}};\n")
    f.write(f"const double MODEL_X_SCALE[{len(x_scale_list)}] = {{{', '.join([f'{v:.8g}' for v in x_scale_list])}}};\n\n")

    f.write(f"const double MODEL_Y_MIN[{len(y_min_list)}]   = {{{', '.join([f'{v:.8g}' for v in y_min_list])}}};\n")
    f.write(f"const double MODEL_Y_SCALE[{len(y_scale_list)}] = {{{', '.join([f'{v:.8g}' for v in y_scale_list])}}};\n")

print(f"Saved: {mqh_path}\nDone ✓")