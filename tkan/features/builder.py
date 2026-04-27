import pandas as pd

from ..data import select_symbol_ohlc
from . import (
    adx,
    bollinger,
    candle_ratios,
    ema_cross,
    garman_klass,
    higher_timeframes,
    log_returns,
    macd,
    price_to_ema,
    price_to_sma,
    rolling_volatility,
    rsi,
    time_features,
    tick_volume,
)

FEATURE_BUILDERS = [
    ('log_returns', log_returns.build),
    ('candle_ratios', candle_ratios.build),
    ('garman_klass', garman_klass.build),
    ('rolling_volatility', rolling_volatility.build),
    ('bollinger', bollinger.build),
    ('price_to_sma', price_to_sma.build),
    ('price_to_ema', price_to_ema.build),
    ('ema_cross', ema_cross.build),
    ('rsi', rsi.build),
    ('adx', adx.build),
    ('macd', macd.build),
    ('higher_timeframes', higher_timeframes.build),
    ('tick_volume', tick_volume.build),
]


def _build_symbol_frame(ohlc, cfg):
    frames = []
    for name, builder in FEATURE_BUILDERS:
        section = cfg['features'][name]
        if not section['enabled']:
            continue
        frame = builder(ohlc, section)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(index=ohlc.index)
    return pd.concat(frames, axis=1)


def build_feature_frame(df, cfg):
    frames = []
    for symbol in cfg['enabled_symbols']:
        frame = _build_symbol_frame(select_symbol_ohlc(df, symbol), cfg)
        if not frame.empty:
            frames.append(frame.add_prefix(f'{symbol}_'))

    if cfg['features']['time']['enabled']:
        index = select_symbol_ohlc(df, cfg['symbol']).index
        frames.append(time_features.build(index, cfg['features']['time']).add_prefix('time_'))

    if not frames:
        raise ValueError('At least one feature must be enabled.')
    return pd.concat(frames, axis=1)
