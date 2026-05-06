import numpy as np
import pandas as pd


def build(index, cfg):
    stamp = index + pd.Timedelta(minutes=1)
    data = {}
    if cfg['hour']:
        data['sin_hour'] = np.sin(2.0 * np.pi * stamp.hour / 24.0)
        data['cos_hour'] = np.cos(2.0 * np.pi * stamp.hour / 24.0)
    if cfg['minute']:
        data['sin_minute'] = np.sin(2.0 * np.pi * stamp.minute / 60.0)
        data['cos_minute'] = np.cos(2.0 * np.pi * stamp.minute / 60.0)
    if cfg['day_of_week']:
        data['day_of_week'] = stamp.dayofweek / 6.0
    return pd.DataFrame(data, index=index)
