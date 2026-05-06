"""
TKAN crypto volume forecasting.
Backend: PyTorch (KERAS_BACKEND=torch)
Exports model.onnx + norm_params.mqh + config.mqh to ./model/

Install deps:
    pip install keras tkan torch onnx pandas pyarrow
"""

import os
import time
import shutil
import numpy as np
import pandas as pd

# ── Must set backend before importing keras ───────────────────────────────────
os.environ.setdefault("KERAS_BACKEND", "torch")

import keras
from keras.models import Sequential
from keras.layers import Input, Dense
from tkan import TKAN

os.makedirs("model", exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
SEQUENCE_LENGTH = 45
N_AHEAD         = 1
BATCH_SIZE      = 128
N_MAX_EPOCHS    = 1000
ASSETS = [
    "BTC", "ETH", "ADA", "XMR", "EOS", "MATIC", "TRX",
    "FTM", "BNB", "XLM", "ENJ", "CHZ", "BUSD", "ATOM",
    "LINK", "ETC", "XRP", "BCH", "LTC",
]

keras.utils.set_random_seed(42)


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


# ── scaler ────────────────────────────────────────────────────────────────────
class MinMaxScaler:
    def __init__(self, feature_axis=None, minmax_range=(0, 1)):
        self.feature_axis = feature_axis
        self.min_ = self.max_ = self.scale_ = None
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
            self.min_ = np.min(X)
            self.max_ = np.max(X)
        self.scale_ = self.max_ - self.min_
        return self

    def transform(self, X):
        X_s = (X - self.min_) / self.scale_
        return X_s * (self.minmax_range[1] - self.minmax_range[0]) + self.minmax_range[0]

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_s):
        X = (X_s - self.minmax_range[0]) / (self.minmax_range[1] - self.minmax_range[0])
        return X * self.scale_ + self.min_


# ── data ──────────────────────────────────────────────────────────────────────
def generate_data(df, sequence_length, n_ahead=1):
    scaler_df = df.copy().shift(n_ahead).rolling(24 * 14).median()
    tmp_df    = df.copy() / scaler_df
    tmp_df    = tmp_df.iloc[24 * 14 + n_ahead:].fillna(0.0)
    scaler_df = scaler_df.iloc[24 * 14 + n_ahead:].fillna(0.0)

    X, y, y_sc = [], [], []
    for i in range(sequence_length, len(tmp_df) - n_ahead + 1):
        X.append(tmp_df.iloc[i - sequence_length:i].values)
        y.append(tmp_df.iloc[i:i + n_ahead, 0:1].values)
        y_sc.append(scaler_df.iloc[i:i + n_ahead, 0:1].values)
    X, y, y_sc = np.array(X), np.array(y), np.array(y_sc)

    split = int(len(X) * 0.8)
    X_tr_raw, X_te_raw = X[:split], X[split:]
    y_tr_raw, y_te_raw = y[:split], y[split:]

    xs = MinMaxScaler(feature_axis=2)
    X_train = xs.fit_transform(X_tr_raw)
    X_test  = xs.transform(X_te_raw)

    ys = MinMaxScaler(feature_axis=2)
    y_train = ys.fit_transform(y_tr_raw).reshape(len(y_tr_raw), -1)
    y_test  = ys.transform(y_te_raw).reshape(len(y_te_raw), -1)

    return xs, X_train, X_test, ys, y_train, y_test


# ── norm / config helpers ─────────────────────────────────────────────────────
def save_norm_params(xmin, xmax, out_dir="model"):
    xmin = np.array(xmin).squeeze()
    xmax = np.array(xmax).squeeze()
    n = len(xmin)
    content = (
        f"const double NORM_MIN[{n}] = {{{', '.join(f'{v:.10g}' for v in xmin)}}};\n"
        f"const double NORM_MAX[{n}] = {{{', '.join(f'{v:.10g}' for v in xmax)}}};\n"
    )
    path = os.path.join(out_dir, "norm_params.mqh")
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved {path}  ({n} features)")


def save_config(n_features, sequence_length, n_ahead, out_dir="model"):
    lines = [
        f"const int    CFG_SEQUENCE_LENGTH = {sequence_length};",
        f"const int    CFG_INPUT_DIM       = {n_features};",
        f"const int    CFG_N_AHEAD         = {n_ahead};",
        f"const double CFG_LOT_SIZE        = 0.01;",
        f"const double CFG_STOP_LOSS_PCT   = 0.02;",
        f"const double CFG_TAKE_PROFIT_PCT = 0.04;",
    ]
    path = os.path.join(out_dir, "config.mqh")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved {path}")


# ── ONNX helpers ──────────────────────────────────────────────────────────────
def _get_shape(vi):
    dims = []
    for d in vi.type.tensor_type.shape.dim:
        if not d.HasField("dim_value"):
            return None
        dims.append(int(d.dim_value))
    return dims


def _set_shape(vi, shape):
    dims = vi.type.tensor_type.shape.dim
    del dims[:]
    for s in shape:
        dims.add().dim_value = int(s)


def _make_mql5_compatible(path):
    """
    Flatten the 3-D input (batch, seq, features) → 2-D (batch, seq*features)
    so MQL5's ONNX runtime can feed a plain float array.
    A Reshape node is prepended inside the graph to restore the 3-D view.
    """
    import onnx
    from onnx import TensorProto, helper

    model = onnx.load(path)
    graph = model.graph
    if not graph.input:
        return path

    mi     = graph.input[0]
    iname  = mi.name
    legacy = f"{iname}_internal"
    ishape = _get_shape(mi)

    # Already 2-D and no legacy input — nothing to do
    if ishape and len(ishape) == 2 and not any(v.name == legacy for v in graph.input[1:]):
        return path

    # Determine the true 3-D shape
    src_shape = ishape if (ishape and len(ishape) == 3) else None
    legacy_shape_inputs = set()
    keep_nodes = []

    for node in graph.node:
        is_legacy_reshape = (
            node.op_type == "Reshape"
            and len(node.input) == 2
            and node.input[0] == iname
            and len(node.output) == 1
            and node.output[0] == legacy
        )
        if is_legacy_reshape:
            legacy_shape_inputs.add(node.input[1])
        else:
            keep_nodes.append(node)

    if src_shape is None:
        for v in graph.input[1:]:
            if v.name == legacy:
                src_shape = _get_shape(v)
                break

    if src_shape is None or len(src_shape) != 3:
        print("Warning: could not patch ONNX for MQL5 compatibility")
        return path

    flat_shape    = [src_shape[0], src_shape[1] * src_shape[2]]
    reshaped_out  = f"{iname}_reshaped"
    shape_const   = "shape_const"

    keep_init = [i for i in graph.initializer if i.name not in legacy_shape_inputs]
    keep_vi   = [v for v in graph.value_info if v.name not in {legacy, *legacy_shape_inputs}]

    shape_tensor = helper.make_tensor("shape_const_value", TensorProto.INT64, [len(src_shape)], src_shape)
    const_node   = helper.make_node("Constant", inputs=[], outputs=[shape_const], value=shape_tensor)
    reshape_node = helper.make_node("Reshape", inputs=[iname, shape_const], outputs=[reshaped_out])

    _set_shape(mi, flat_shape)
    del graph.input[1:]
    del graph.initializer[:]
    graph.initializer.extend(keep_init)
    del graph.value_info[:]
    graph.value_info.extend(keep_vi)

    fixed = []
    for node in keep_nodes:
        updated = [reshaped_out if n in (iname, legacy) else n for n in node.input]
        del node.input[:]
        node.input.extend(updated)
        fixed.append(node)

    del graph.node[:]
    graph.node.extend([const_node, reshape_node, *fixed])

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f"Saved MQL5-compatible ONNX: {path}")
    return path


def export_onnx_torch(model, sequence_length, n_features, out_dir="model"):
    """
    Export a Keras-on-PyTorch model to ONNX using torch.onnx.export.

    Strategy:
      1. Extract the underlying torch.nn.Module from the Keras model.
      2. Trace it with a dummy input.
      3. Export via torch.onnx and patch for MQL5 compatibility.
    """
    import torch

    path = os.path.join(out_dir, "model.onnx")

    # Build a dummy input on the same device as the model's first parameter.
    try:
        device = next(model.layers[1].weights[0].value.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    dummy = torch.zeros(1, sequence_length, n_features, dtype=torch.float32, device=device)

    # Keras 3 (torch backend) exposes the raw nn.Module via model._tracker or
    # by simply calling model(x) — we wrap the Keras call in a thin nn.Module
    # so torch.onnx can trace it cleanly.
    class KerasWrapper(torch.nn.Module):
        def __init__(self, keras_model):
            super().__init__()
            self.km = keras_model
            # Register all torch sub-modules so their parameters are visible.
            for i, layer in enumerate(keras_model.layers):
                if hasattr(layer, "_tracker"):
                    for j, mod in enumerate(layer._tracker.dict["module"].values()):
                        if isinstance(mod, torch.nn.Module):
                            self.add_module(f"layer_{i}_{j}", mod)

        def forward(self, x):
            import torch
            # Keras forward; disable training-mode behaviours.
            return torch.tensor(
                self.km(x.numpy(), training=False).numpy()
            )

    # Prefer a simpler approach: call model.predict on dummy to warm up, then
    # use torch.onnx.export with the Keras model directly as a callable.
    # Keras 3's torch backend makes model() a valid torch.nn.Module callable.
    torch_module = model  # Keras Sequential IS a torch.nn.Module under torch backend

    # Warm-up so all lazy layers are built.
    with torch.no_grad():
        _ = torch_module(dummy)

    torch.onnx.export(
        torch_module,
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported via torch.onnx → {path}")
    _make_mql5_compatible(path)
    return path


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    # --- load data ---
    df = pd.read_parquet("data.parquet")
    df = df[
        (df.index >= pd.Timestamp("2020-01-01")) &
        (df.index <  pd.Timestamp("2023-01-01"))
    ]
    df = df[
        [c for c in df.columns
         if "quote asset volume" in c and any(a in c for a in ASSETS)]
    ]
    df.columns = [c.replace(" quote asset volume", "") for c in df.columns]
    cols = ["BTC"] + [c for c in df.columns if c != "BTC"]
    df   = df[cols]
    print(f"Data shape: {df.shape}  columns: {list(df.columns)}")

    n_features = df.shape[1]

    # --- prepare sequences ---
    xs, X_train, X_test, ys, y_train, y_test = generate_data(df, SEQUENCE_LENGTH, N_AHEAD)

    # --- save norm params ---
    save_norm_params(xs.min_, xs.max_, out_dir="model")
    save_config(n_features, SEQUENCE_LENGTH, N_AHEAD, out_dir="model")
    print(f"X_train: {X_train.shape}  y_train: {y_train.shape}")

    # --- build model ---
    model = Sequential([
        Input(shape=(SEQUENCE_LENGTH, n_features)),
        TKAN(100, return_sequences=True),
        TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
        Dense(units=N_AHEAD, activation="linear"),
    ], name="TKAN")
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss="mse",
        jit_compile=False,   # TorchScript/torch.compile — set True if supported
    )
    model.summary()

    # --- train ---
    t0 = time.time()
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=N_MAX_EPOCHS,
        validation_split=0.2,
        callbacks=make_callbacks(),
        shuffle=True,
        verbose=1,
    )
    print(f"Training time: {time.time() - t0:.1f}s")

    # --- export ONNX (torch-native, no TF) ---
    try:
        import torch   # guaranteed present since KERAS_BACKEND=torch
        export_onnx_torch(model, SEQUENCE_LENGTH, n_features, out_dir="model")
    except Exception as e:
        print(f"torch.onnx export failed: {e}")
        print("Model weights are still available via model.save_weights('model/weights.weights.h5')")
        model.save_weights("model/weights.weights.h5")

    print("\nDone. Files in ./model/:")
    for f in sorted(os.listdir("model")):
        print(" ", f)


if __name__ == "__main__":
    main()