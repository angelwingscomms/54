import pandas as pd

from .utils import rsi


def build(ohlc, cfg):
    close = ohlc['close']
    data = {f'rsi_{period}': rsi(close, period) for period in cfg['periods']}
    return pd.DataFrame(data, index=ohlc.index)
