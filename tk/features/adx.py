import pandas as pd

from .utils import adx


def build(ohlc, cfg):
    data = {f'adx_{period}': adx(ohlc, period) for period in cfg['periods']}
    return pd.DataFrame(data, index=ohlc.index)
