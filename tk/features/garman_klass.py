import numpy as np
import pandas as pd

from .utils import safe_log_ratio


def build(ohlc, cfg):
    log_hl = safe_log_ratio(ohlc['high'], ohlc['low'])
    log_co = safe_log_ratio(ohlc['close'], ohlc['open'])
    variance = 0.5 * log_hl.pow(2) - (2.0 * np.log(2.0) - 1.0) * log_co.pow(2)
    variance = variance.clip(lower=0.0)
    data = {
        f'gk_vol_{window}': variance.rolling(window, min_periods=window).mean().pow(0.5)
        for window in cfg['windows']
    }
    return pd.DataFrame(data, index=ohlc.index)
