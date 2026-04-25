import yaml

DEFAULT_FEATURE_SYMBOLS = [
    'BCHUSD', 'BTCUSD', 'ETHUSD', 'LTCUSD', 'XRPUSD', 'ADAUSD', 'AVAXUSD',
    'AXSUSD', 'DOGEUSD', 'DOTUSD', 'EOSUSD', 'FILUSD', 'LINKUSD', 'MATICUSD',
    'MIOTAUSD', 'SOLUSD', 'TRXUSD', 'UNIUSD', 'XLMUSD',
]

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
    'learning_rate': 0.01,
    'epochs': 1,
    'train_test_split': 0.8,
    'feature_symbols': {symbol: True for symbol in DEFAULT_FEATURE_SYMBOLS},
}


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
    print("\n" + "="*50)
    print("LOADING CONFIG")
    print("="*50)
    with open('config.yaml') as f:
        raw_cfg = yaml.safe_load(f) or {}
    cfg = {**DEFAULTS, **raw_cfg}
    cfg['feature_symbols'] = {
        **DEFAULTS['feature_symbols'],
        **(raw_cfg.get('feature_symbols') or {}),
    }
    cfg = resolve_feature_symbols(cfg)
    print(f"  config.yaml loaded successfully")
    print(f"  Config keys found: {list(raw_cfg.keys())}")
    print("-"*50)
    print("  Applied settings:")
    for k, v in DEFAULTS.items():
        if k in raw_cfg:
            print(f"    [{k}] = {cfg[k]} (from config)")
        else:
            print(f"    [{k}] = {v} (DEFAULT)")
    print(f"    [enabled_symbols] = {cfg['enabled_symbols']}")
    print("="*50 + "\n")
    return cfg
