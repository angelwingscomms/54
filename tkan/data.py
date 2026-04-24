import pandas as pd
from tkan.config import load_config


def load_data(config_path='config.yaml'):
    cfg = load_config(config_path)
    df = pd.read_parquet(cfg['data_path'])
    ohlc_cols = []
    for asset in cfg['assets']:
        for pt in ['open', 'high', 'low', 'close']:
            col = f"{asset} {pt}"
            if col in df.columns:
                ohlc_cols.append(col)
    if not ohlc_cols:
        raise ValueError("No OHLC columns matched — check asset names vs data columns")
    df = df[ohlc_cols]
    df = df[(df.index >= pd.Timestamp(cfg['start_date'])) & (df.index < pd.Timestamp(cfg['end_date']))]
    return cfg, df
