import yaml

DEFAULTS = {
    'symbol': 'BTC',
    'target_type': 'atr',
    'atr_multiplier': 2.0,
    'tp_multiplier': 2.0,
    'atr_period': 9,
    'threshold_pct': 1.0,
    'stop_loss_pct': 0.5,
    'n_ahead': 9,
    'data_path': 'data.csv',
}


def load_config():
    print("\n" + "="*50)
    print("LOADING CONFIG")
    print("="*50)
    with open('config.yaml') as f:
        cfg = yaml.safe_load(f) or {}
    print(f"  config.yaml loaded successfully")
    print(f"  Config keys found: {list(cfg.keys())}")
    print("-"*50)
    print("  Applied settings:")
    for k, v in DEFAULTS.items():
        if k in cfg:
            print(f"    [{k}] = {v} (from config)")
        else:
            print(f"    [{k}] = {v} (DEFAULT)")
    print("="*50 + "\n")
    return {**DEFAULTS, **cfg}