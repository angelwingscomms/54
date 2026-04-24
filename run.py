import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import jax
import jax.numpy as jnp
import numpy as np
import time
import optax
from jax2onnx import to_onnx
from tkan.data import load_data
from tkan.labels import create_binary_labels, split_train_test, normalize_features, save_norm_params
from tkan.tkan_model import init_tkan, tkan_apply, binary_crossentropy, compute_accuracy, make_apply_fn

cfg, df = load_data()
seq_len = cfg['sequence_length']
n_ahead = cfg['n_ahead']
threshold = cfg['threshold_pct']
stop_loss = cfg['stop_loss_pct']
hidden = cfg['hidden_size']
sub = cfg['sub_dim']
bs = cfg['batch_size']
lr = cfg['learning_rate']
epochs = cfg['epochs']
split = cfg['train_test_split']
seed = cfg['seed']

print(f"Data: {df.shape}, TF={cfg['timeframe_minutes']}m, TP={threshold}%, SL={stop_loss}%, n_ahead={n_ahead}")
X, y = create_binary_labels(df, cfg)
print(f"Labels: 1={int(np.sum(y))}, 0={int(len(y)-np.sum(y))}, pos_rate={np.mean(y)*100:.1f}%")

X_tr, X_te, y_tr, y_te = split_train_test(X, y, split)
X_tr, X_te, xmin, xmax = normalize_features(X_tr, X_te)
save_norm_params(xmin, xmax, cfg['norm_output'])
y_tr, y_te = y_tr.reshape(-1, 1), y_te.reshape(-1, 1)
X_tr, y_tr, X_te, y_te = map(jnp.array, [X_tr, y_tr, X_te, y_te])

input_dim = X_tr.shape[-1]
print(f"Train: {X_tr.shape}, Test: {X_te.shape}, input_dim={input_dim}")

key = jax.random.key(seed)
key, k = jax.random.split(key)
params = init_tkan(input_dim, hidden, sub, k)
print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

opt = optax.adam(lr)
opt_st = opt.init(params)

start = time.time()
for ep in range(epochs):
    key, sk = jax.random.split(key)
    idx = jax.random.permutation(sk, len(X_tr))
    ep_loss, ep_acc, n_b = 0.0, 0.0, 0
    for i in range(0, len(X_tr), bs):
        bx, by = X_tr[idx[i:i+bs]], y_tr[idx[i:i+bs]]
        def loss_fn(p):
            return binary_crossentropy(tkan_apply(p, bx, use_sigmoid=True), by)
        l, g = jax.value_and_grad(loss_fn)(params)
        u, opt_st = opt.update(g, opt_st)
        params = optax.apply_updates(params, u)
        ep_loss += l
        ep_acc += compute_accuracy(tkan_apply(params, bx, use_sigmoid=True), by)
        n_b += 1
    print(f"  Epoch {ep+1}: loss={ep_loss/n_b:.4f} acc={ep_acc/n_b:.4f}")

preds = tkan_apply(params, X_te, use_sigmoid=True)
print(f"Time: {time.time()-start:.1f}s, Test Acc: {compute_accuracy(preds, y_te):.4f}")

print("Exporting ONNX...")
result = to_onnx(
    make_apply_fn(params, use_sigmoid=True),
    inputs=[jax.ShapeDtypeStruct((1, seq_len, input_dim), jnp.float32)],
    model_name='TKAN', return_mode='file', output_path=cfg['model_output']
)
print(f"Saved {cfg['model_output']}: {result}")
