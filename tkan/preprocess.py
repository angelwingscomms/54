import pandas as pd
import numpy as np


def compute_atr(btc, period=9):
    prev_close = btc['close'].shift(1)
    tr = np.maximum(
        btc['high'] - btc['low'],
        np.maximum(
            np.abs(btc['high'] - prev_close),
            np.abs(btc['low'] - prev_close)
        )
    )
    atr = tr.rolling(window=period).mean()
    return atr


def build_samples(btc, atr, horizon, tp_pct, tolerance,
                 target_type, atr_multiplier, tp_multiplier):
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
            sl_price = close * (1 + tolerance / 100)

        sl_pct = -tp_pct * tolerance
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