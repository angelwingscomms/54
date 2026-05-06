import pandas as pd

from .utils import safe_div, sma


def build(ohlc, cfg):
    close = ohlc['close']
    data = {}
    for period in cfg['periods']:
        mean = sma(close, period)
        std = close.rolling(period, min_periods=period).std(ddof=0)
        upper = mean + cfg['std'] * std
        lower = mean - cfg['std'] * std
        data[f'bb_width_{period}'] = safe_div(upper - lower, mean)
        data[f'bb_pctb_{period}'] = safe_div(close - lower, upper - lower)
    return pd.DataFrame(data, index=ohlc.index)
