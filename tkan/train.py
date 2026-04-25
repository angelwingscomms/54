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
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    num_batches = len(range(0, len(X_tr), 128))
    best_params = params
    best_val_loss = float('inf')

    for ep in range(epochs):
        ep_start = time.time()
        idx = jax.random.permutation(jax.random.key(ep), len(X_tr))
        ep_loss = 0
        
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
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
        
        print(f"Epoch {ep+1:2d}/{epochs} | val_loss: {val_loss:.4f} | val_acc: {100*val_acc:.2f}% | train_loss: {train_loss:.4f} | train_acc: {100*train_acc:.2f}%")

    elapsed = time.time() - start
    best_val_preds = tkan_apply(best_params, X_va)
    acc = jnp.mean((best_val_preds > 0.5) == y_va)
    print(f"\nDone! Time: {elapsed:.1f}s | Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {100*acc:.2f}%")

    return best_params, train_losses, val_losses, train_accs, val_accs, elapsed
