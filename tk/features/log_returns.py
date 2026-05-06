import pandas as pd

from .utils import log_return


def build(ohlc, cfg):
    close = ohlc['close']
    data = {f'log_return_{period}': log_return(close, period) for period in cfg['periods']}
    return pd.DataFrame(data, index=ohlc.index)
