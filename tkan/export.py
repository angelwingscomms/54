import jax
from jax2onnx import to_onnx
import jax.numpy as jnp
import onnx
from onnx import TensorProto, helper


def save_norm_params(xmin, xmax):
    xmin = xmin.squeeze()
    xmax = xmax.squeeze()
    n = len(xmin)
    min_str = ", ".join(f"{v:.10g}" for v in xmin)
    max_str = ", ".join(f"{v:.10g}" for v in xmax)
    content = (
        f"const double NORM_MIN[{n}] = {{{min_str}}};\n"
        f"const double NORM_MAX[{n}] = {{{max_str}}};\n"
    )
    with open("norm_params.mqh", "w") as f:
        f.write(content)
    print(f"Saved norm_params.mqh ({n} features)")


def save_config(cfg):
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
    ]
    for key, mql_type, name in mappings:
        v = cfg.get(key)
        if v is None:
            from .config import DEFAULTS
            v = DEFAULTS.get(key)
        if mql_type == 'string':
            content.append(f'const string {name} = "{v}";')
        else:
            content.append(f'const {mql_type} {name} = {v};')

    feature_symbols = cfg.get('enabled_symbols') or [cfg.get('symbol')]
    input_dim = cfg.get('input_dim', len(feature_symbols) * 4)
    content.append(f'const string CFG_FEATURE_SYMBOLS = "{",".join(feature_symbols)}";')
    content.append(f'const int CFG_INPUT_DIM = {input_dim};')

    with open('config.mqh', 'w') as f:
        f.write('\n'.join(content))
    print('Saved config.mqh')


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


def to_onnx_model(params, sequence_length=45, input_dim=4, hidden=100, sub=20):
    from .tkan_forward import tkan_fwd

    def make_apply_fn(params_inner):
        def apply_fn(x):
            return jax.nn.sigmoid(jnp.dot(tkan_fwd(params_inner, x, hidden), params_inner['dense_w']) + params_inner['dense_b'])
        return apply_fn

    result = to_onnx(
        make_apply_fn(params),
        inputs=[jax.ShapeDtypeStruct((1, sequence_length, input_dim), jnp.float32)],
        model_name='TKAN',
        return_mode='file',
        output_path='model.onnx'
    )
    make_mql5_compatible('model.onnx')
    return result
