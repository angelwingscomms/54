import pandas as pd

OHLC_FIELDS = ('open', 'high', 'low', 'close', 'tick_volume')


def load_csv(path):
    if not path.endswith('.csv'):
        path += '.csv'
    df = pd.read_csv(path, encoding='utf-16', parse_dates=['datetime'], date_format='%Y-%m-%d %H-%M')
    if 'symbol' not in df.columns:
        return df.set_index('datetime').sort_index()

    wide = df.pivot(index='datetime', columns='symbol', values=list(OHLC_FIELDS)).sort_index()
    wide.columns = [f'{symbol}_{field}' for field, symbol in wide.columns]
    return wide.sort_index()


def ohlc_columns(symbol):
    return [f'{symbol}_{field}' for field in OHLC_FIELDS]


def select_symbol_ohlc(df, symbol):
    cols = ohlc_columns(symbol)
    if all(col in df.columns for col in cols):
        out = df[cols].copy()
        out.columns = OHLC_FIELDS
        return out
    if all(field in df.columns for field in OHLC_FIELDS):
        return df[list(OHLC_FIELDS)].copy()
    missing = [col for col in cols if col not in df.columns]
    raise ValueError(f'Missing target columns for {symbol}: {missing}')


def select_feature_frame(df, symbols):
    cols = []
    for symbol in symbols:
        cols.extend(ohlc_columns(symbol))

    if all(col in df.columns for col in cols):
        return df[cols].copy()
    if len(symbols) == 1 and all(field in df.columns for field in OHLC_FIELDS):
        return df[list(OHLC_FIELDS)].copy()
    missing = [col for col in cols if col not in df.columns]
    raise ValueError(f'Missing feature columns: {missing}')
