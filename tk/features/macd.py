import pandas as pd

from .utils import normalized_macd_hist


def build(ohlc, cfg):
    name = f'macd_hist_{cfg["fast"]}_{cfg["slow"]}_{cfg["signal"]}'
    return pd.DataFrame(
        {name: normalized_macd_hist(ohlc['close'], cfg['fast'], cfg['slow'], cfg['signal'])},
        index=ohlc.index,
    )
