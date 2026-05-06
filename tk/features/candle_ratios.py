import pandas as pd

from .utils import safe_div


def build(ohlc, cfg):
    del cfg
    return pd.DataFrame(
        {
            'close_open_ratio': safe_div(ohlc['close'], ohlc['open']) - 1.0,
            'high_low_ratio': safe_div(ohlc['high'], ohlc['low']) - 1.0,
        },
        index=ohlc.index,
    )
