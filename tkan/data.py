import pandas as pd


def load_csv(path):
    return pd.read_csv(path, index_col=0, parse_dates=True, encoding='utf-16', date_format='%Y-%m-%d %H-%M').sort_index()
