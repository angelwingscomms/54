import jax
import jax.numpy as jnp
import optax
import time
from .tkan_init import init_tkan
from .tkan_apply import tkan_apply
from .loss import bce_loss, eval_loss


def train(X_tr, y_tr, X_va, y_va, input_dim, hidden=100, sub=20, epochs=27, lr=1e-3):
    key = jax.random.key(42)
    key, k = jax.random.split(key)
    params = init_tkan(input_dim, hidden, sub, k)
    print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    opt = optax.adam(lr)
    opt_st = opt.init(params)

    start = time.time()
    train_losses, val_losses = [], []
    num_batches = len(range(0, len(X_tr), 128))

    for ep in range(epochs):
        ep_start = time.time()
        idx = jax.random.permutation(jax.random.key(ep), len(X_tr))
        ep_loss = 0
        
        pct = (ep + 1) / epochs
        bar_len = 20
        filled = int(bar_len * pct)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\rEpoch {ep+1:2d}/{epochs} [{bar}] {pct*100:.1f}%", end="", flush=True)
        
        for i in range(0, len(X_tr), 128):
            b_idx = idx[i:i+128]
            bx, by = X_tr[b_idx], y_tr[b_idx]
            l, g = jax.value_and_grad(bce_loss)(params, bx, by)
            u, opt_st = opt.update(g, opt_st)
            params = optax.apply_updates(params, u)
            ep_loss += l

        train_loss = float(ep_loss) / num_batches
        val_loss = eval_loss(params, X_va, y_va)
        
        train_preds = tkan_apply(params, X_tr)
        train_acc = float(jnp.mean((train_preds > 0.5) == y_tr))
        val_preds = tkan_apply(params, X_va)
        val_acc = float(jnp.mean((val_preds > 0.5) == y_va))
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        ep_elapsed = time.time() - ep_start
        eta = ep_elapsed * (epochs - ep - 1)
        pct = (ep + 1) * 100 // epochs
        print(f"Epoch {ep+1}/{epochs} [{pct}%] | train: {train_loss:.4f} ({100*train_acc:.1f}%) | val: {val_loss:.4f} ({100*val_acc:.1f}%) | ETA: {eta:.0f}s")

    elapsed = time.time() - start
    preds = tkan_apply(params, X_va)
    acc = jnp.mean((preds > 0.5) == y_va)
    print(f"Time: {elapsed:.1f}s, Validation Accuracy: {acc:.4f}")

    return params, train_losses, val_losses, acc, elapsed
