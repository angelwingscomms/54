import pandas as pd

from .utils import safe_div, sma


def build(ohlc, cfg):
    close = ohlc['close']
    data = {
        f'price_to_sma_{period}': safe_div(close, sma(close, period)) - 1.0
        for period in cfg['periods']
    }
    return pd.DataFrame(data, index=ohlc.index)
