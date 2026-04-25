import jax
import jax.numpy as jnp
import optax
import time
from .tkan_init import init_tkan
from .tkan_apply import tkan_apply
from .loss import bce_loss, eval_loss


def train(X_tr, y_tr, X_te, y_te, input_dim, hidden=100, sub=20, epochs=27, lr=1e-3):
    key = jax.random.key(42)
    key, k = jax.random.split(key)
    params = init_tkan(input_dim, hidden, sub, k)
    print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    opt = optax.adam(lr)
    opt_st = opt.init(params)

    start = time.time()
    train_losses, val_losses = [], []

    for ep in range(epochs):
        idx = jax.random.permutation(jax.random.key(ep), len(X_tr))
        ep_loss = 0
        for i in range(0, len(X_tr), 128):
            b_idx = idx[i:i+128]
            bx, by = X_tr[b_idx], y_tr[b_idx]
            l, g = jax.value_and_grad(bce_loss)(params, bx, by)
            u, opt_st = opt.update(g, opt_st)
            params = optax.apply_updates(params, u)
            ep_loss += l

        num_batches = len(range(0, len(X_tr), 128))
        train_loss = float(ep_loss) / num_batches
        val_loss = eval_loss(params, X_te, y_te)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        if (ep+1) % 2 == 0:
            print(f"  Epoch {ep+1}: train={train_loss:.4f}  val={val_loss:.4f}")

    elapsed = time.time() - start
    preds = tkan_apply(params, X_te)
    acc = jnp.mean((preds > 0.5) == y_te)
    print(f"Time: {elapsed:.1f}s, Accuracy: {acc:.4f}")

    return params, train_losses, val_losses, acc, elapsed