import pandas as pd

from .utils import ema, safe_div


def build(ohlc, cfg):
    close = ohlc['close']
    cache = {}
    data = {}
    for fast, slow in cfg['pairs']:
        if fast not in cache:
            cache[fast] = ema(close, fast)
        if slow not in cache:
            cache[slow] = ema(close, slow)
        data[f'ema_cross_{fast}_{slow}'] = safe_div(cache[fast], cache[slow]) - 1.0
    return pd.DataFrame(data, index=ohlc.index)
