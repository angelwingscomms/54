import yaml
from pathlib import Path

REQUIRED = [
    'data_path', 'timeframe_minutes', 'sequence_length', 'n_ahead', 'assets',
    'start_date', 'end_date', 'threshold_pct', 'stop_loss_pct',
    'hidden_size', 'sub_dim', 'batch_size', 'learning_rate', 'epochs',
    'train_test_split', 'seed', 'model_output', 'norm_output',
]


def load_config(path='config.yaml'):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with open(p) as f:
        cfg = yaml.safe_load(f)
    for k in REQUIRED:
        if k not in cfg:
            raise ValueError(f"Missing config key: {k}")
    if not isinstance(cfg['assets'], list) or len(cfg['assets']) == 0:
        raise ValueError("'assets' must be a non-empty list")
    if not (0 < cfg['train_test_split'] < 1):
        raise ValueError("'train_test_split' must be between 0 and 1")
    if cfg['timeframe_minutes'] <= 0:
        raise ValueError("'timeframe_minutes' must be positive")
    return cfg
