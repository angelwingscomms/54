import numpy as np


def compute_atr(ohlc, period=9):
    prev_close = ohlc['close'].shift(1)
    tr = np.maximum(
        ohlc['high'] - ohlc['low'],
        np.maximum(
            np.abs(ohlc['high'] - prev_close),
            np.abs(ohlc['low'] - prev_close)
        )
    )
    atr = tr.rolling(window=period).mean()
    return atr


def build_samples(features, target, atr, sequence_length, horizon, tp_pct, tolerance,
                 target_type, atr_multiplier, tp_multiplier):
    X, y = [], []
    for i in range(sequence_length, len(features) - horizon):
        X.append(features.iloc[i - sequence_length:i].values)

        close = target.iloc[i]['close']

        if target_type == 'atr':
            atr_val = atr.iloc[i]
            sl_distance = atr_val * atr_multiplier
            tp_distance = sl_distance * tp_multiplier
        else:
            tp_distance = close * (tp_pct / 100)
            sl_distance = close * (tolerance / 100)

        long_tp = close + tp_distance
        long_sl = close - sl_distance
        short_tp = close - tp_distance
        short_sl = close + sl_distance

        long_tp_idx = long_sl_idx = short_tp_idx = short_sl_idx = None
        for j in range(1, horizon + 1):
            high = target.iloc[i + j]['high']
            low = target.iloc[i + j]['low']
            if long_tp_idx is None and high >= long_tp:
                long_tp_idx = j
            if long_sl_idx is None and low <= long_sl:
                long_sl_idx = j
            if short_tp_idx is None and low <= short_tp:
                short_tp_idx = j
            if short_sl_idx is None and high >= short_sl:
                short_sl_idx = j
            if long_tp_idx and long_sl_idx and short_tp_idx and short_sl_idx:
                break

        hit_up = min(
            long_tp_idx if long_tp_idx else horizon + 1,
            short_sl_idx if short_sl_idx else horizon + 1
        )
        hit_down = min(
            short_tp_idx if short_tp_idx else horizon + 1,
            long_sl_idx if long_sl_idx else horizon + 1
        )

        y.append(1.0 if hit_up < hit_down else 0.0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)
