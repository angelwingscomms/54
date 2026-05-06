import pandas as pd

from .utils import align_completed_frame, completed_resample, log_return, normalized_macd_hist, rsi


def build(ohlc, cfg):
    frames = []
    for minutes in cfg['timeframes']:
        higher = completed_resample(ohlc, minutes)
        close = higher['close']
        data = {}
        for period in cfg['log_return_periods']:
            data[f'htf_{minutes}_log_return_{period}'] = log_return(close, period)
        for period in cfg['rsi_periods']:
            data[f'htf_{minutes}_rsi_{period}'] = rsi(close, period)
        macd_cfg = cfg['macd']
        data[f'htf_{minutes}_macd_hist_{macd_cfg["fast"]}_{macd_cfg["slow"]}_{macd_cfg["signal"]}'] = (
            normalized_macd_hist(close, macd_cfg['fast'], macd_cfg['slow'], macd_cfg['signal'])
        )
        frames.append(align_completed_frame(pd.DataFrame(data, index=higher.index), ohlc.index))
    if not frames:
        return pd.DataFrame(index=ohlc.index)
    return pd.concat(frames, axis=1)
