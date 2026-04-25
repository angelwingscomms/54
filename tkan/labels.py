import numpy as np


def get_trade_label(current_price, future_prices, threshold_pct, stop_loss_pct):
    tp = current_price * (1 + threshold_pct / 100)
    sl = current_price * (1 - stop_loss_pct / 100)
    for price in future_prices:
        if price <= sl:
            return 0
        if price >= tp:
            return 1
    return 0


def create_binary_labels(df, cfg):
    seq = cfg['sequence_length']
    na = cfg['n_ahead']
    btc_close = 'BTC close'
    X = np.array([df.iloc[i - seq:i].values for i in range(seq, len(df) - na)], dtype=np.float32)
    y = np.array([
        get_trade_label(df.iloc[i][btc_close], df.iloc[i+1:i+1+na][btc_close].values,
                        cfg['threshold_pct'], cfg['stop_loss_pct'])
        for i in range(seq, len(df) - na)
    ], dtype=np.float32)
    return X, y


def split_train_test(X, y, ratio):
    s = int(len(X) * ratio)
    return X[:s], X[s:], y[:s], y[s:]


def normalize_features(X_train, X_test):
    xmin = X_train.min(axis=(0, 1))
    xmax = X_train.max(axis=(0, 1))
    X_train = (X_train - xmin) / (xmax - xmin + 1e-8)
    X_test = (X_test - xmin) / (xmax - xmin + 1e-8)
    return X_train, X_test, xmin, xmax


def save_norm_params(xmin, xmax, path='norm_params.mqh'):
    n = len(xmin)
    with open(path, 'w') as f:
        f.write(f'const double NORM_MIN[{n}] = {{{",".join(f"{v:.10f}" for v in xmin)}}};\n')
        f.write(f'const double NORM_MAX[{n}] = {{{",".join(f"{v:.10f}" for v in xmax)}}};\n')
