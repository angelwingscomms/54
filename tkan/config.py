import copy

import yaml

DEFAULT_FEATURE_SYMBOLS = [
    'BCHUSD', 'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'ADAUSD', 'AVAXUSD',
    'AXSUSD', 'DOGEUSD', 'DOTUSD', 'EOSUSD', 'FILUSD', 'LINKUSD', 'MATICUSD',
    'MIOTAUSD', 'SOLUSD', 'TRXUSD', 'UNIUSD', 'XLMUSD',
]

def _default_features():
    return {
        'log_returns': {'enabled': True, 'periods': [1, 5, 15, 60]},
        'candle_ratios': {'enabled': True},
        'garman_klass': {'enabled': True, 'windows': [5, 15, 60]},
        'rolling_volatility': {'enabled': True, 'windows': [5, 15, 60]},
        'bollinger': {'enabled': True, 'periods': [20], 'std': 2.0},
        'price_to_sma': {'enabled': True, 'periods': [15]},
        'price_to_ema': {'enabled': True, 'periods': [5, 15, 50]},
        'ema_cross': {'enabled': True, 'pairs': [[5, 15], [15, 50]]},
        'rsi': {'enabled': True, 'periods': [7, 14]},
        'adx': {'enabled': True, 'periods': [14]},
        'macd': {'enabled': True, 'fast': 12, 'slow': 26, 'signal': 9},
        'higher_timeframes': {
            'enabled': True,
            'timeframes': [15, 60],
            'log_return_periods': [1],
            'rsi_periods': [14],
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
        },
        'time': {'enabled': True, 'hour': True, 'minute': True, 'day_of_week': True},
        'tick_volume': {'enabled': True, 'timeframes': [15, 60]},
    }


DEFAULTS = {
    'symbol': 'BTCUSD',
    'target_type': 'atr',
    'atr_multiplier': 2.0,
    'tp_multiplier': 2.0,
    'atr_period': 9,
    'threshold_pct': 1.0,
    'stop_loss_pct': 0.5,
    'n_ahead': 9,
    'data_path': 'data.csv',
    'sequence_length': 45,
    'hidden_size': 100,
    'sub_dim': 20,
    'batch_size': 128,
    'learning_rate': 0.01,
    'epochs': 1,
    'seed': 42,
    'train_test_split': 0.8,
    'confidence_threshold': 0.6,
    'model_output': 'model.onnx',
    'norm_output': 'norm_params.mqh',
    'feature_symbols': {symbol: True for symbol in DEFAULT_FEATURE_SYMBOLS},
    'features': _default_features(),
}


def _merge_dict(base, extra):
    out = copy.deepcopy(base)
    for key, value in (extra or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _int_list(values):
    return [int(v) for v in values]


def _pair_list(values):
    pairs = []
    for fast, slow in values:
        fast = int(fast)
        slow = int(slow)
        if fast >= slow:
            raise ValueError(f'EMA cross pair must be fast < slow, got {fast}, {slow}')
        pairs.append([fast, slow])
    return pairs


def normalize_feature_config(raw_features):
    features = _merge_dict(_default_features(), raw_features)
    features['log_returns']['periods'] = _int_list(features['log_returns']['periods'])
    features['garman_klass']['windows'] = _int_list(features['garman_klass']['windows'])
    features['rolling_volatility']['windows'] = _int_list(features['rolling_volatility']['windows'])
    features['bollinger']['periods'] = _int_list(features['bollinger']['periods'])
    features['bollinger']['std'] = float(features['bollinger']['std'])
    features['price_to_sma']['periods'] = _int_list(features['price_to_sma']['periods'])
    features['price_to_ema']['periods'] = _int_list(features['price_to_ema']['periods'])
    features['ema_cross']['pairs'] = _pair_list(features['ema_cross']['pairs'])
    features['rsi']['periods'] = _int_list(features['rsi']['periods'])
    features['adx']['periods'] = _int_list(features['adx']['periods'])
    for key in ('fast', 'slow', 'signal'):
        features['macd'][key] = int(features['macd'][key])
        features['higher_timeframes']['macd'][key] = int(features['higher_timeframes']['macd'][key])
    features['higher_timeframes']['timeframes'] = _int_list(features['higher_timeframes']['timeframes'])
    features['higher_timeframes']['log_return_periods'] = _int_list(features['higher_timeframes']['log_return_periods'])
    features['higher_timeframes']['rsi_periods'] = _int_list(features['higher_timeframes']['rsi_periods'])
    return features


def resolve_feature_symbols(cfg):
    toggles = {symbol: True for symbol in DEFAULT_FEATURE_SYMBOLS}
    toggles.update(cfg.get('feature_symbols') or {})

    target = cfg['symbol']
    if not toggles.get(target, False):
        print(f"  forcing feature_symbols[{target}] = true because it is the target symbol")
        toggles[target] = True

    order = DEFAULT_FEATURE_SYMBOLS + [symbol for symbol in toggles if symbol not in DEFAULT_FEATURE_SYMBOLS]
    enabled = [symbol for symbol in order if toggles.get(symbol)]
    if not enabled:
        raise ValueError('At least one feature symbol must be enabled.')

    cfg['feature_symbols'] = {symbol: bool(toggles[symbol]) for symbol in order}
    cfg['enabled_symbols'] = enabled
    return cfg


def load_config():
    print("="*50)
    print("LOADING CONFIG")
    print("="*50)
    with open('config.yaml') as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = {**DEFAULTS, **{k: v for k, v in raw_cfg.items() if k != 'features'}}
    cfg['feature_symbols'] = {
        **DEFAULTS['feature_symbols'],
        **(raw_cfg.get('feature_symbols') or {}),
    }
    cfg['features'] = normalize_feature_config(raw_cfg.get('features'))
    cfg = resolve_feature_symbols(cfg)
    print(f"  config.yaml loaded successfully")
    print("-"*50)
    print("feature_symbols:")
    for symbol, enabled in cfg['feature_symbols'].items():
        print(f"  {symbol}: {enabled}")
    print("features:")
    for group, section in cfg['features'].items():
        print(f"  {group}:")
        for k, v in section.items():
            print(f"    {k}: {v}")
    print("enabled_symbols:")
    for symbol in cfg['enabled_symbols']:
        print(f"  - {symbol}")
    print("enabled_feature_groups:")
    for name in [n for n, s in cfg['features'].items() if s.get('enabled')]:
        print(f"  - {name}")
    print("="*50 + "\n")
    return cfg
