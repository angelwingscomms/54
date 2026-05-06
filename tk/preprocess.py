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


def _resolve_trade(future, tp_price, sl_price, is_buy):
    for row in future.itertuples(index=False):
        tp_hit = row.high >= tp_price if is_buy else row.low <= tp_price
        sl_hit = row.low <= sl_price if is_buy else row.high >= sl_price
        if tp_hit and sl_hit:
            return 'ambiguous'
        if tp_hit:
            return 'tp'
        if sl_hit:
            return 'sl'
    return None


def build_samples(features, target, atr, sequence_length, horizon, tp_pct, tolerance,
                 target_type, atr_multiplier, tp_multiplier):
    X, y = [], []
    for i in range(sequence_length - 1, len(features) - horizon):
        close = float(target.iloc[i]['close'])

        if target_type == 'atr':
            atr_val = float(atr.iloc[i])
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            sl_distance = atr_val * atr_multiplier
            tp_distance = sl_distance * tp_multiplier
        else:
            tp_distance = close * (tp_pct / 100.0)
            sl_distance = close * ((tp_pct * tolerance) / 100.0)

        if tp_distance <= 0 or sl_distance <= 0:
            continue

        future = target.iloc[i + 1:i + horizon + 1]
        long_result = _resolve_trade(future, close + tp_distance, close - sl_distance, True)
        short_result = _resolve_trade(future, close - tp_distance, close + sl_distance, False)

        if 'ambiguous' in (long_result, short_result):
            continue

        long_win = long_result == 'tp'
        short_win = short_result == 'tp'
        if long_win == short_win:
            continue

        X.append(features.iloc[i - sequence_length + 1:i + 1].values)
        y.append(1.0 if long_win else 0.0)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32).reshape(-1, 1)
