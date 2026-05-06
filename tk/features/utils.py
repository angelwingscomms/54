import numpy as np
import pandas as pd

EPS = 1e-12


def safe_denominator(series):
    return series.mask(series.abs() < EPS)


def safe_div(numerator, denominator):
    return numerator / safe_denominator(denominator)


def safe_log_ratio(numerator, denominator):
    return np.log(safe_div(numerator, denominator))


def log_return(close, period):
    return safe_log_ratio(close, close.shift(period))


def sma(series, period):
    return series.rolling(period, min_periods=period).mean()


def ema(series, period):
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(close, period):
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss + EPS)
    return 100.0 - (100.0 / (1.0 + rs))


def adx(ohlc, period):
    high = ohlc['high']
    low = ohlc['low']
    close = ohlc['close']
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=ohlc.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=ohlc.index,
    )
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean() / (atr + EPS)
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di + EPS)
    return dx.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def macd_hist(close, fast, slow, signal):
    macd = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd, signal)
    return macd - signal_line


def normalized_macd_hist(close, fast, slow, signal):
    return safe_div(macd_hist(close, fast, slow, signal), close)


def completed_resample(ohlc, minutes):
    shifted = ohlc.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=1)
    rule = f'{int(minutes)}min'
    aggs = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
    if 'tick_volume' in ohlc.columns:
        aggs['tick_volume'] = 'sum'
    return shifted.resample(rule, label='right', closed='right').agg(aggs).dropna()


def align_completed_frame(frame, target_index, has_tick_volume=False):
    aligned = frame.reindex(target_index + pd.Timedelta(minutes=1), method='ffill')
    aligned.index = target_index
    return aligned
