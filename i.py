"""
train.py — TKAN multi-symbol cryptocurrency / forex trading model
=================================================================
* Connects to MetaTrader 5 to fetch historical hourly OHLCV data
* Trains a two-layer TKAN network (no TensorFlow — torch backend)
* Exports the trained model to ONNX (model.onnx)
* Saves per-feature normalisation params (scaler_params.csv + .json)

Dependencies (install once):
    pip install MetaTrader5 tkan keras torch onnx scikit-learn numpy pandas

IMPORTANT
  – Copy model.onnx  →  <MT5 data folder>/MQL5/Files/
  – Copy scaler_params.csv → same folder (or terminal's common Files/)
"""

import os, json, time
import numpy as np
import pandas as pd

# ── Use PyTorch backend — NO TensorFlow ──────────────────────────────────────
os.environ["KERAS_BACKEND"] = "torch"

import torch
import keras
from keras.models import Sequential
from keras.layers import Dense, Input
from tkan import TKAN

import MetaTrader5 as mt5
from sklearn.metrics import r2_score, root_mean_squared_error

# ── Configuration ─────────────────────────────────────────────────────────────

# All 21 symbols requested (broker symbol names may vary — edit if needed)
SYMBOLS: list[str] = [
    "$USDX",   "USDJPY",  "BCHUSD",  "BTCUSD",  "ETHUSD",
    "LTCUSD",  "XRPUSD",  "ADAUSD",  "AVAXUSD", "AXSUSD",
    "DOGEUSD", "DOTUSD",  "EOSUSD",  "FILUSD",  "LINKUSD",
    "MATICUSD","MIOTAUSD","SOLUSD",  "TRXUSD",  "UNIUSD",
    "XLMUSD",
]

PRIMARY_SYMBOL  = "BTCUSD"   # symbol the EA will trade
TIMEFRAME       = mt5.TIMEFRAME_H1
N_BARS          = 26_304      # ~3 years of hourly bars
SEQUENCE_LEN    = 45          # look-back window
N_AHEAD         = 1           # bars ahead to predict
BATCH_SIZE      = 128
MAX_EPOCHS      = 1_000

OUT_ONNX        = "model.onnx"
OUT_SCALER_CSV  = "scaler_params.csv"
OUT_SCALER_JSON = "scaler_params.json"

keras.utils.set_random_seed(42)


# ── MT5 helpers ───────────────────────────────────────────────────────────────

def init_mt5() -> None:
    if not mt5.initialize():
        raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")
    info = mt5.terminal_info()
    print(f"MT5 connected: {info.name}  build={info.build}")


def fetch_closes(symbols: list[str]) -> pd.DataFrame:
    """
    Pull close prices for every symbol; unavailable symbols are filled
    with NaN then replaced with 0 so the model always sees N=21 features.
    """
    series: dict[str, pd.Series] = {}
    for sym in symbols:
        rates = mt5.copy_rates_from_pos(sym, TIMEFRAME, 0, N_BARS)
        if rates is None or len(rates) < SEQUENCE_LEN + 100:
            n = 0 if rates is None else len(rates)
            print(f"  [skip] {sym:>12s}  ({n} bars — unavailable / too short)")
            series[sym] = pd.Series(dtype=float, name=sym)
            continue
        df = pd.DataFrame(rates)[["time", "close"]]
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        series[sym] = df["close"].rename(sym)
        print(f"  [ok]   {sym:>12s}  ({len(df)} bars)")

    combined = pd.DataFrame(series)          # aligns on common timestamps
    combined.sort_index(inplace=True)

    # Fill unavailable symbols with 0 so we keep exactly len(symbols) cols
    combined.fillna(0.0, inplace=True)

    # Drop rows where ALL values are 0 (no data at all for that timestamp)
    combined = combined[(combined != 0).any(axis=1)]
    return combined[symbols]                 # enforce original column order


# ── Preprocessing ─────────────────────────────────────────────────────────────

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log-return; rows where price is 0 yield 0 (not NaN) via masking."""
    p = prices.replace(0.0, np.nan)
    lr = np.log(p / p.shift(1))
    lr.fillna(0.0, inplace=True)
    return lr.iloc[1:]                       # drop first NaN row


class MinMaxScaler:
    """Feature-wise [0, 1] scaler fitted only on training data."""

    def __init__(self):
        self.min_: np.ndarray | None = None
        self.max_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        self.min_ = X.min(axis=0)
        self.max_ = X.max(axis=0)
        return self

    def _scale(self) -> np.ndarray:
        sc = self.max_ - self.min_
        sc[sc == 0] = 1.0                    # avoid division by zero
        return sc

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / self._scale()

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def make_sequences(
    data: np.ndarray, target_col: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window: X → [batch, SEQUENCE_LEN, features], y → [batch, N_AHEAD]."""
    X, y = [], []
    for i in range(SEQUENCE_LEN, len(data) - N_AHEAD + 1):
        X.append(data[i - SEQUENCE_LEN : i])
        y.append(data[i : i + N_AHEAD, target_col])
    return (np.asarray(X, dtype=np.float32),
            np.asarray(y, dtype=np.float32))


# ── Model ─────────────────────────────────────────────────────────────────────

def build_tkan(n_features: int) -> keras.Model:
    model = Sequential(
        [
            Input(shape=(SEQUENCE_LEN, n_features)),
            TKAN(100, return_sequences=True),
            TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20,
                 return_sequences=False),
            Dense(N_AHEAD, activation="linear"),
        ],
        name="TKAN",
    )
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="mean_squared_error")
    model.summary()
    return model


def callbacks() -> list:
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


# ── ONNX export (torch backend → torch.onnx) ──────────────────────────────────

def export_onnx(model: keras.Model, n_features: int) -> None:
    """Wrap the Keras/torch model and export via torch.onnx."""

    class _Wrapper(torch.nn.Module):
        def __init__(self, km: keras.Model):
            super().__init__()
            self.km = km

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.km(x, training=False)

    wrapper = _Wrapper(model).eval()
    dummy   = torch.zeros(1, SEQUENCE_LEN, n_features, dtype=torch.float32)

    # --- Primary: standard export -------------------------------------------
    try:
        torch.onnx.export(
            wrapper, dummy, OUT_ONNX,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=17,
        )
        print(f"✓ ONNX saved  →  {OUT_ONNX}")
        return
    except Exception as err:
        print(f"  torch.onnx.export failed ({err}); trying dynamo_export …")

    # --- Fallback: dynamo-based export (torch ≥ 2.1) ------------------------
    try:
        export_prog = torch.onnx.dynamo_export(wrapper, dummy)
        export_prog.save(OUT_ONNX)
        print(f"✓ ONNX (dynamo) saved  →  {OUT_ONNX}")
    except Exception as err2:
        raise RuntimeError(
            f"ONNX export failed entirely: {err2}\n"
            "Try:  pip install onnx onnxscript  and upgrade torch≥2.1"
        ) from err2


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── 1. Fetch data ────────────────────────────────────────────────────────
    print("\n=== Connecting to MetaTrader 5 ===")
    init_mt5()

    print("\n=== Fetching hourly closes ===")
    closes = fetch_closes(SYMBOLS)
    mt5.shutdown()

    n_features = len(SYMBOLS)                 # always 21
    target_col = SYMBOLS.index(PRIMARY_SYMBOL)

    print(f"\nDataset:  {closes.shape[0]} bars × {n_features} symbols")
    print(f"Period:   {closes.index[0]}  →  {closes.index[-1]}")
    avail = [s for s in SYMBOLS if closes[s].any()]
    print(f"Non-zero: {len(avail)} / {n_features} symbols")

    # ── 2. Log-returns & normalisation ───────────────────────────────────────
    returns   = log_returns(closes)           # (T, 21)
    data      = returns.values.astype(np.float32)

    split     = int(len(data) * 0.80)
    scaler    = MinMaxScaler()
    train_sc  = scaler.fit_transform(data[:split])
    test_sc   = scaler.transform(data[split:])

    # ── 3. Save scaler params ────────────────────────────────────────────────
    # JSON (rich format)
    meta = {
        "symbols":        SYMBOLS,
        "primary_symbol": PRIMARY_SYMBOL,
        "target_col":     target_col,
        "sequence_len":   SEQUENCE_LEN,
        "n_features":     n_features,
        "min":            scaler.min_.tolist(),
        "max":            scaler.max_.tolist(),
    }
    with open(OUT_SCALER_JSON, "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"✓ {OUT_SCALER_JSON}")

    # CSV — two rows: [0]=min values, [1]=max values  (no header)
    # MQL5 reads these as two sequential lines of comma-separated floats
    np.savetxt(OUT_SCALER_CSV,
               np.vstack([scaler.min_, scaler.max_]),
               delimiter=",", fmt="%.12f", comments="")
    print(f"✓ {OUT_SCALER_CSV}")

    # ── 4. Sequences ─────────────────────────────────────────────────────────
    X_train, y_train = make_sequences(train_sc, target_col)

    # Build test sequences: last SEQUENCE_LEN rows of train provide history
    full_sc = np.vstack([train_sc, test_sc])
    overlap  = full_sc[split - SEQUENCE_LEN :]
    X_all, y_all = make_sequences(overlap, target_col)
    X_test = X_all
    y_test = y_all

    print(f"\nTrain sequences: {X_train.shape}")
    print(f"Test  sequences: {X_test.shape}")

    # ── 5. Train ─────────────────────────────────────────────────────────────
    print("\n=== Training TKAN (torch backend, no TensorFlow) ===")
    model = build_tkan(n_features)

    t0 = time.time()
    model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=MAX_EPOCHS,
        validation_split=0.20,
        callbacks=callbacks(),
        shuffle=True,
        verbose=1,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s")

    # ── 6. Evaluate ──────────────────────────────────────────────────────────
    preds = model.predict(X_test, verbose=0)
    r2   = r2_score(y_test, preds)
    rmse = root_mean_squared_error(y_test, preds)
    print(f"\nTest  R²  : {r2:.4f}")
    print(f"Test RMSE : {rmse:.8f}")

    # ── 7. ONNX export ───────────────────────────────────────────────────────
    print("\n=== Exporting to ONNX ===")
    export_onnx(model, n_features)

    print("\n=== Done ===")
    print(f"Copy to MT5 Files/:  {OUT_ONNX}  +  {OUT_SCALER_CSV}")


if __name__ == "__main__":
    main()