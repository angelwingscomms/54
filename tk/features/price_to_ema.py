import pandas as pd

from .utils import ema, safe_div


def build(ohlc, cfg):
    close = ohlc['close']
    data = {
        f'price_to_ema_{period}': safe_div(close, ema(close, period)) - 1.0
        for period in cfg['periods']
    }
    return pd.DataFrame(data, index=ohlc.index)
