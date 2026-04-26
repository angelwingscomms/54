import yaml
from pathlib import Path


def save_model_outputs(
    model_dir,
    cfg,
    params,
    xmin,
    xmax,
    seq_len,
    input_dim,
    hidden,
    sub,
    train_losses,
    val_losses,
    train_accs,
    val_accs,
    best_epoch,
    best_val_loss,
    best_val_acc,
    test_loss,
    test_acc,
    elapsed,
    update_live_mq5=True,
):
    from .export import save_norm_params, save_config, to_onnx_model

    save_norm_params(xmin, xmax, output_dir=str(model_dir))
    save_config(cfg, output_dir=str(model_dir))

    with open(model_dir / 'config.yaml', 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"Saved config.yaml to {model_dir}")

    notes = f"""# Model Training Notes

## Best Epoch
- **Epoch**: {best_epoch} (of {cfg['epochs']} total)

## Training Metrics (at best epoch)
- **Train Loss**: {train_losses[best_epoch-1]:.6f}
- **Train Accuracy**: {train_accs[best_epoch-1]*100:.2f}%
- **Val Loss**: {val_losses[best_epoch-1]:.6f}
- **Val Accuracy**: {val_accs[best_epoch-1]*100:.2f}%

## Final Metrics (after best epoch)
- **Best Val Loss**: {best_val_loss:.6f}
- **Best Val Accuracy**: {best_val_acc*100:.2f}%
- **Test Loss**: {test_loss:.6f}
- **Test Accuracy**: {test_acc*100:.2f}%

## Training Time
- **Total elapsed**: {elapsed:.1f} seconds

## Configuration
- **Sequence length**: {cfg['sequence_length']}
- **Hidden size**: {cfg['hidden_size']}
- **Sub dim**: {cfg['sub_dim']}
- **Batch size**: {cfg['batch_size']}
- **Learning rate**: {cfg['learning_rate']}
- **Seed**: {cfg['seed']}
- **Input dim**: {input_dim}

## Epoch History
"""
    for ep in range(cfg['epochs']):
        notes += f"- Epoch {ep+1}: train_loss={train_losses[ep]:.6f}, train_acc={train_accs[ep]*100:.2f}%, val_loss={val_losses[ep]:.6f}, val_acc={val_accs[ep]*100:.2f}%\n"

    with open(model_dir / 'notes.md', 'w') as f:
        f.write(notes)
    print(f"Saved notes.md to {model_dir}")

    print("\nExporting model to ONNX...")
    to_onnx_model(params, sequence_length=seq_len, input_dim=input_dim, hidden=hidden, sub=sub, output_dir=str(model_dir))

    if update_live_mq5:
        update_live_mq5_paths(model_dir)


def update_live_mq5_paths(model_dir):
    ts = model_dir.name
    live_mq5 = Path('live.mq5')
    if live_mq5.exists():
        content = live_mq5.read_text()
        content = content.replace('#include "config.mqh"', f'#include "models/{ts}/config.mqh"')
        content = content.replace('#include "norm_params.mqh"', f'#include "models/{ts}/norm_params.mqh"')
        content = content.replace('#resource "\\\\Experts\\\\TKAN\\\\model.onnx"', f'#resource "\\\\Experts\\\\54\\\\models\\\\{ts}\\\\model.onnx"')
        live_mq5.write_text(content)
        print(f"Updated live.mq5 to use model: {ts}")