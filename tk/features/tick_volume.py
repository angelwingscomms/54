import numpy as np
import pandas as pd

from .utils import align_completed_frame, completed_resample


def build(ohlc, cfg):
    frames = []
    tick_col = 'tick_volume'

    if tick_col not in ohlc.columns:
        return pd.DataFrame(index=ohlc.index)

    data = {}
    raw_vol = ohlc[tick_col].copy()
    data[f'tick_volume'] = raw_vol
    data[f'log_tick_volume'] = np.log1p(raw_vol)
    frames.append(pd.DataFrame(data, index=ohlc.index))

    for minutes in cfg['timeframes']:
        higher = completed_resample(ohlc, minutes)
        if tick_col not in higher.columns:
            continue
        higher_vol = higher[tick_col]
        htf_data = {}
        htf_data[f'htf_{minutes}_tick_volume_sum'] = higher_vol
        htf_data[f'htf_{minutes}_tick_volume_mean'] = higher_vol / minutes
        htf_data[f'htf_{minutes}_log_tick_volume'] = np.log1p(higher_vol)
        frames.append(align_completed_frame(pd.DataFrame(htf_data, index=higher.index), ohlc.index))

    if not frames:
        return pd.DataFrame(index=ohlc.index)
    return pd.concat(frames, axis=1)