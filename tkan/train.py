import jax
import jax.numpy as jnp
import optax
import time
from .tkan_init import init_tkan
from .tkan_apply import tkan_apply
from .loss import bce_loss, eval_loss


def train(X_tr, y_tr, X_va, y_va, X_te, y_te, input_dim, hidden=100, sub=20, epochs=27, lr=1e-3, batch_size=128, seed=42):
    key = jax.random.key(seed)
    key, k = jax.random.split(key)
    params = init_tkan(input_dim, hidden, sub, k)
    print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    opt = optax.adam(lr)
    opt_st = opt.init(params)

    start = time.time()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    num_batches = len(range(0, len(X_tr), batch_size))
    best_params = params
    best_val_loss = float('inf')
    best_epoch = 0

    for ep in range(epochs):
        ep_start = time.time()
        idx = jax.random.permutation(jax.random.key(seed + ep), len(X_tr))
        ep_loss = 0
        
        for i in range(0, len(X_tr), batch_size):
            b_idx = idx[i:i+batch_size]
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
            best_epoch = ep + 1
        
        

    elapsed = time.time() - start
    best_val_preds = tkan_apply(best_params, X_va)
    best_val_acc = float(jnp.mean((best_val_preds > 0.5) == y_va))
    test_loss = float(eval_loss(best_params, X_te, y_te))
    test_preds = tkan_apply(best_params, X_te)
    test_acc = float(jnp.mean((test_preds > 0.5) == y_te))
    print(f"epoch {best_epoch} | val_loss {best_val_loss:.4f} | val_acc {100*best_val_acc:.2f}% | test_loss {test_loss:.4f} | test_acc {100*test_acc:.2f}% | time {elapsed:.1f}s")

    return best_params, train_losses, val_losses, train_accs, val_accs, elapsed
