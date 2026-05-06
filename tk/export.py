import jax
from jax2onnx import to_onnx
import jax.numpy as jnp
import onnx
from onnx import TensorProto, helper
from pathlib import Path


def save_norm_params(xmean, xstd, output_dir='.'):
    xmean = xmean.squeeze()
    xstd = xstd.squeeze()
    n = len(xmean)
    mean_str = ", ".join(f"{v:.10g}" for v in xmean)
    std_str = ", ".join(f"{v:.10g}" for v in xstd)
    content = (
        f"const double NORM_MEAN[{n}] = {{{mean_str}}};\n"
        f"const double NORM_STD[{n}] = {{{std_str}}};\n"
    )
    path = Path(output_dir) / "norm_params.mqh"
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved norm_params.mqh ({n} features) to {path.parent}")


def save_norm_params_regression(xmin, xmax, output_dir='.'):
    xmin = xmin.squeeze()
    xmax = xmax.squeeze()
    n = len(xmin)
    min_str = ", ".join(f"{v:.10g}" for v in xmin)
    max_str = ", ".join(f"{v:.10g}" for v in xmax)
    content = (
        f"const double NORM_MIN[{n}] = {{{min_str}}};\n"
        f"const double NORM_MAX[{n}] = {{{max_str}}};\n"
    )
    path = Path(output_dir) / "norm_params.mqh"
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved norm_params.mqh ({n} features) to {path.parent}")


def save_config(cfg, output_dir='.'):
    def fmt(value):
        if isinstance(value, bool):
            return 'true' if value else 'false'
        if isinstance(value, str):
            return f'"{value}"'
        if isinstance(value, float):
            return f'{value:.10g}'
        return str(value)

    def add_scalar(name, mql_type, value):
        content.append(f'const {mql_type} {name} = {fmt(value)};')

    def add_array(name, mql_type, values, empty_value):
        values = list(values)
        if not values:
            values = [empty_value]
        content.append(f'const {mql_type} {name}[] = {{{", ".join(fmt(v) for v in values)}}};')

    content = []
    mappings = [
        ('symbol', 'string', 'CFG_SYMBOL'),
        ('target_type', 'string', 'CFG_TARGET_TYPE'),
        ('atr_multiplier', 'double', 'CFG_ATR_MULTIPLIER'),
        ('tp_multiplier', 'double', 'CFG_TP_MULTIPLIER'),
        ('atr_period', 'int', 'CFG_ATR_PERIOD'),
        ('threshold_pct', 'double', 'CFG_THRESHOLD_PCT'),
        ('stop_loss_pct', 'double', 'CFG_TOLERANCE'),
        ('sequence_length', 'int', 'CFG_SEQUENCE_LENGTH'),
        ('confidence_threshold', 'double', 'CFG_CONFIDENCE_THRESHOLD'),
    ]
    for key, mql_type, name in mappings:
        v = cfg.get(key)
        if v is None:
            from .config import DEFAULTS
            v = DEFAULTS.get(key)
        add_scalar(name, mql_type, v)

    feature_symbols = cfg.get('enabled_symbols') or [cfg.get('symbol')]
    input_dim = cfg.get('input_dim', len(feature_symbols) * 4)
    add_scalar('CFG_FEATURE_SYMBOLS', 'string', ','.join(feature_symbols))
    add_scalar('CFG_INPUT_DIM', 'int', input_dim)

    features = cfg['features']
    add_scalar('CFG_LOG_RETURNS_ENABLED', 'bool', features['log_returns']['enabled'])
    add_array('CFG_LOG_RETURN_PERIODS', 'int', features['log_returns']['periods'], 0)
    add_scalar('CFG_CANDLE_RATIOS_ENABLED', 'bool', features['candle_ratios']['enabled'])
    add_scalar('CFG_GARMAN_KLASS_ENABLED', 'bool', features['garman_klass']['enabled'])
    add_array('CFG_GARMAN_KLASS_WINDOWS', 'int', features['garman_klass']['windows'], 0)
    add_scalar('CFG_ROLLING_VOLATILITY_ENABLED', 'bool', features['rolling_volatility']['enabled'])
    add_array('CFG_ROLLING_VOLATILITY_WINDOWS', 'int', features['rolling_volatility']['windows'], 0)
    add_scalar('CFG_BOLLINGER_ENABLED', 'bool', features['bollinger']['enabled'])
    add_array('CFG_BOLLINGER_PERIODS', 'int', features['bollinger']['periods'], 0)
    add_scalar('CFG_BOLLINGER_STD', 'double', features['bollinger']['std'])
    add_scalar('CFG_PRICE_TO_SMA_ENABLED', 'bool', features['price_to_sma']['enabled'])
    add_array('CFG_PRICE_TO_SMA_PERIODS', 'int', features['price_to_sma']['periods'], 0)
    add_scalar('CFG_PRICE_TO_EMA_ENABLED', 'bool', features['price_to_ema']['enabled'])
    add_array('CFG_PRICE_TO_EMA_PERIODS', 'int', features['price_to_ema']['periods'], 0)
    add_scalar('CFG_EMA_CROSS_ENABLED', 'bool', features['ema_cross']['enabled'])
    add_array('CFG_EMA_CROSS_FAST', 'int', [pair[0] for pair in features['ema_cross']['pairs']], 0)
    add_array('CFG_EMA_CROSS_SLOW', 'int', [pair[1] for pair in features['ema_cross']['pairs']], 0)
    add_scalar('CFG_RSI_ENABLED', 'bool', features['rsi']['enabled'])
    add_array('CFG_RSI_PERIODS', 'int', features['rsi']['periods'], 0)
    add_scalar('CFG_ADX_ENABLED', 'bool', features['adx']['enabled'])
    add_array('CFG_ADX_PERIODS', 'int', features['adx']['periods'], 0)
    add_scalar('CFG_MACD_ENABLED', 'bool', features['macd']['enabled'])
    add_scalar('CFG_MACD_FAST', 'int', features['macd']['fast'])
    add_scalar('CFG_MACD_SLOW', 'int', features['macd']['slow'])
    add_scalar('CFG_MACD_SIGNAL', 'int', features['macd']['signal'])
    add_scalar('CFG_HIGHER_TIMEFRAMES_ENABLED', 'bool', features['higher_timeframes']['enabled'])
    add_array('CFG_HIGHER_TIMEFRAME_MINUTES', 'int', features['higher_timeframes']['timeframes'], 0)
    add_array('CFG_HIGHER_TIMEFRAME_LOG_RETURN_PERIODS', 'int', features['higher_timeframes']['log_return_periods'], 0)
    add_array('CFG_HIGHER_TIMEFRAME_RSI_PERIODS', 'int', features['higher_timeframes']['rsi_periods'], 0)
    add_scalar('CFG_HIGHER_TIMEFRAME_MACD_FAST', 'int', features['higher_timeframes']['macd']['fast'])
    add_scalar('CFG_HIGHER_TIMEFRAME_MACD_SLOW', 'int', features['higher_timeframes']['macd']['slow'])
    add_scalar('CFG_HIGHER_TIMEFRAME_MACD_SIGNAL', 'int', features['higher_timeframes']['macd']['signal'])
    add_scalar('CFG_TIME_FEATURES_ENABLED', 'bool', features['time']['enabled'])
    add_scalar('CFG_TIME_HOUR_ENABLED', 'bool', features['time']['hour'])
    add_scalar('CFG_TIME_MINUTE_ENABLED', 'bool', features['time']['minute'])
    add_scalar('CFG_TIME_DAY_OF_WEEK_ENABLED', 'bool', features['time']['day_of_week'])

    path = Path(output_dir) / 'config.mqh'
    with open(path, 'w') as f:
        f.write('\n'.join(content))
    print(f'Saved config.mqh to {path.parent}')


def _get_shape(value_info):
    dims = []
    for dim in value_info.type.tensor_type.shape.dim:
        if not dim.HasField('dim_value'):
            return None
        dims.append(int(dim.dim_value))
    return dims


def _set_shape(value_info, shape):
    dims = value_info.type.tensor_type.shape.dim
    del dims[:]
    for size in shape:
        dims.add().dim_value = int(size)


def make_mql5_compatible(path='model.onnx'):
    model = onnx.load(path)
    graph = model.graph
    if not graph.input:
        return path

    model_input = graph.input[0]
    input_name = model_input.name
    legacy_name = f'{input_name}_internal'
    has_legacy_input = any(value.name == legacy_name for value in graph.input[1:])
    input_shape = _get_shape(model_input)

    if input_shape and len(input_shape) == 2 and not has_legacy_input:
        return path

    source_shape = input_shape if input_shape and len(input_shape) == 3 else None
    legacy_shape_inputs = set()
    keep_nodes = []

    for node in graph.node:
        is_legacy_reshape = (
            node.op_type == 'Reshape'
            and len(node.input) == 2
            and node.input[0] == input_name
            and len(node.output) == 1
            and node.output[0] == legacy_name
        )
        if is_legacy_reshape:
            legacy_shape_inputs.add(node.input[1])
            continue
        keep_nodes.append(node)

    if source_shape is None:
        for value in graph.input[1:]:
            if value.name == legacy_name:
                source_shape = _get_shape(value)
                break

    if source_shape is None:
        raise ValueError('Cannot infer original ONNX input shape for MQL5 compatibility patch.')
    if len(source_shape) != 3:
        raise ValueError(f'Expected 3D ONNX input before flatten patch, got {source_shape}.')

    flat_shape = [source_shape[0], source_shape[1] * source_shape[2]]
    reshape_output = f'{input_name}_reshaped'
    shape_const_name = 'shape_const'
    keep_initializers = [init for init in graph.initializer if init.name not in legacy_shape_inputs]
    keep_value_info = [
        value for value in graph.value_info
        if value.name != legacy_name and value.name not in legacy_shape_inputs
    ]
    shape_tensor = helper.make_tensor(
        name='shape_const_value',
        data_type=TensorProto.INT64,
        dims=[len(source_shape)],
        vals=source_shape,
    )
    const_node = helper.make_node('Constant', inputs=[], outputs=[shape_const_name], value=shape_tensor)
    reshape_node = helper.make_node('Reshape', inputs=[input_name, shape_const_name], outputs=[reshape_output])

    _set_shape(model_input, flat_shape)

    del graph.input[1:]
    del graph.initializer[:]
    graph.initializer.extend(keep_initializers)
    del graph.value_info[:]
    graph.value_info.extend(keep_value_info)

    fixed_nodes = []
    for node in keep_nodes:
        updated_inputs = [reshape_output if name in (input_name, legacy_name) else name for name in node.input]
        del node.input[:]
        node.input.extend(updated_inputs)
        fixed_nodes.append(node)

    del graph.node[:]
    graph.node.extend([const_node, reshape_node, *fixed_nodes])

    onnx.checker.check_model(model)
    onnx.save(model, path)
    print(f'Saved MQL5-compatible ONNX: {path}')
    return path


def save_config_regression(cfg, output_dir='.'):
    def fmt(value):
        if isinstance(value, float):
            return f'{value:.10g}'
        return str(value)

    def add_scalar(name, mql_type, value):
        content.append(f'const {mql_type} {name} = {fmt(value)};')

    content = []
    add_scalar('CFG_SYMBOL', 'string', cfg.get('symbol', ''))
    add_scalar('CFG_SEQUENCE_LENGTH', 'int', cfg.get('sequence_length', 54))
    add_scalar('CFG_INPUT_DIM', 'int', cfg.get('input_dim', 3))

    symbols = cfg.get('enabled_symbols', [])
    if not symbols:
        symbols = [cfg.get('symbol', 'XAUUSD')]
    add_scalar('CFG_FEATURE_SYMBOLS', 'string', ','.join(symbols))

    path = Path(output_dir) / 'config.mqh'
    with open(path, 'w') as f:
        f.write('\n'.join(content))
    print(f'Saved regression config.mqh to {path.parent}')


def to_onnx_regression(params, sequence_length=45, input_dim=4, hidden=100, sub=20, output_dir='.'):
    from .tkan_forward import tkan_fwd

    def make_apply_fn(params_inner):
        def apply_fn(x):
            return jnp.dot(tkan_fwd(params_inner, x, hidden, sub), params_inner['dense_w']) + params_inner['dense_b']
        return apply_fn

    output_path = str(Path(output_dir) / 'model.onnx')
    to_onnx(
        make_apply_fn(params),
        inputs=[jax.ShapeDtypeStruct((1, sequence_length, input_dim), jnp.float32)],
        model_name='TKAN',
        return_mode='file',
        output_path=output_path
    )
    make_mql5_compatible(output_path)
    print(f'Model saved to: {output_path}')
    return output_path


def to_onnx_model(params, sequence_length=45, input_dim=4, hidden=100, sub=20, output_dir='.'):
    from .tkan_forward import tkan_fwd

    def make_apply_fn(params_inner):
        def apply_fn(x):
            return jax.nn.sigmoid(jnp.dot(tkan_fwd(params_inner, x, hidden, sub), params_inner['dense_w']) + params_inner['dense_b'])
        return apply_fn

    output_path = str(Path(output_dir) / 'model.onnx')
    result = to_onnx(
        make_apply_fn(params),
        inputs=[jax.ShapeDtypeStruct((1, sequence_length, input_dim), jnp.float32)],
        model_name='TKAN',
        return_mode='file',
        output_path=output_path
    )
    make_mql5_compatible(output_path)
    print(f'Model saved to: {output_path}')
    return result
