import pandas as pd

from .utils import log_return


def build(ohlc, cfg):
    returns = log_return(ohlc['close'], 1)
    data = {
        f'rolling_vol_{window}': returns.rolling(window, min_periods=window).std(ddof=0)
        for window in cfg['windows']
    }
    return pd.DataFrame(data, index=ohlc.index)
