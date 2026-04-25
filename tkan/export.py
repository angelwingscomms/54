from jax2onnx import to_onnx
import jax.numpy as jnp


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

    with open('config.mqh', 'w') as f:
        f.write('\n'.join(content))
    print('Saved config.mqh')


def to_onnx_model(params, hidden=100, sub=20):
    from .tkan_forward import tkan_fwd

    def make_apply_fn(params_inner):
        def apply_fn(x):
            return jax.nn.sigmoid(jnp.dot(tkan_fwd(params_inner, x, hidden), params_inner['dense_w']) + params_inner['dense_b'])
        return apply_fn

    return to_onnx(
        make_apply_fn(params),
        inputs=[jax.ShapeDtypeStruct((1, 45, 4), jnp.float32)],
        model_name='TKAN',
        return_mode='file',
        output_path='model.onnx'
    )