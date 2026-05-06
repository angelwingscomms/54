import jax
import jax.numpy as jnp
import optax
import time
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.console import Console
from .tkan_init import init_tkan
from .tkan_apply import tkan_apply
from .loss import bce_loss, eval_loss


def _decay_mask(params):
    return {
        'wx': True,
        'uh': True,
        'bias': False,
        'sub_wx': True,
        'sub_wh': True,
        'sub_k': False,
        'agg_w': True,
        'agg_b': False,
        'dense_w': True,
        'dense_b': False,
    }


def train(X_tr, y_tr, X_va, y_va, X_te, y_te, input_dim, hidden=100, sub=20, epochs=27, lr=1e-3, batch_size=128, seed=42):
    console = Console()
    num_batches = len(range(0, len(X_tr), batch_size))
    key = jax.random.key(seed)
    key, k = jax.random.split(key)
    params = init_tkan(input_dim, hidden, sub, k)
    print(f"Params: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

    total_steps = max(1, num_batches * epochs)
    warmup_steps = max(1, int(0.05 * total_steps))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-6,
        peak_value=lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=1e-6,
    )
    opt = optax.adamw(
        learning_rate=schedule,
        weight_decay=1e-4,
        mask=_decay_mask,
    )
    opt_st = opt.init(params)

    start = time.time()
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_params = params
    best_val_loss = float('inf')
    best_epoch = 0

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=20, complete_style="green"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"[green]Epochs", total=epochs)

        for ep in range(epochs):
            batch_task = progress.add_task("[green]Batches", total=num_batches)
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
                progress.update(batch_task, advance=1)

            progress.remove_task(batch_task)
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
            print(f"epoch {ep+1} | valloss {val_loss:.4f} | valacc {val_acc:.4f} | trainloss {train_loss:.4f} | trainacc {train_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = params
                best_epoch = ep + 1
            
            progress.update(task, advance=1, description=f"[green]Epoch {ep+1}/{epochs}")

    elapsed = time.time() - start
    best_val_preds = tkan_apply(best_params, X_va)
    best_val_acc = float(jnp.mean((best_val_preds > 0.5) == y_va))
    test_loss = float(eval_loss(best_params, X_te, y_te))
    test_preds = tkan_apply(best_params, X_te)
    test_acc = float(jnp.mean((test_preds > 0.5) == y_te))
    print(f"epoch {best_epoch} | val_loss {best_val_loss:.4f} | val_acc {100*best_val_acc:.2f}% | test_loss {test_loss:.4f} | test_acc {100*test_acc:.2f}% | time {elapsed:.1f}s")

    final_train_loss = float(bce_loss(best_params, X_tr, y_tr) / len(X_tr) * batch_size)
    final_train_preds = tkan_apply(best_params, X_tr)
    final_train_acc = float(jnp.mean((final_train_preds > 0.5) == y_tr))

    return best_params, train_losses, val_losses, train_accs, val_accs, elapsed, best_epoch, best_val_loss, best_val_acc, test_loss, test_acc, final_train_loss, final_train_acc, test_preds
