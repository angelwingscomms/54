"""
TKAN Training Script
Trains a Temporal Kernel Attention Network for binary classification (buy/sell signals).
"""
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

# ============================================================================
# CONFIGURATION: Load settings from config file
# ============================================================================
config, df = load_data()

# Extract hyperparameters for convenience
sequence_length = config['sequence_length']  # How many time steps to look back
n_ahead = config['n_ahead']                    # How far into the future to predict
threshold_pct = config['threshold_pct']       # Min price change % to trigger BUY signal
stop_loss_pct = config['stop_loss_pct']       # Stop loss % for label generation
hidden_size = config['hidden_size']           # Number of hidden units in TKAN
sub_dim = config['sub_dim']                    # Subdimension for low-rank approximation
batch_size = config['batch_size']             # Mini-batch size for training
learning_rate = config['learning_rate']        # Adam optimizer learning rate
epochs = config['epochs']                       # Number of training epochs
train_test_split = config['train_test_split'] # Fraction of data for training (e.g., 0.8 = 80%)
seed = config['seed']                          # Random seed for reproducibility

# ============================================================================
# DATA PREPARATION: Load data and create labels
# ============================================================================
print(f"Data shape: {df.shape}, Timeframe: {config['timeframe_minutes']}m, Threshold: {threshold_pct}%, Stop Loss: {stop_loss_pct}%, Predict {n_ahead} bars ahead")

# Create binary labels: 1 = BUY signal (price goes up enough), 0 = HOLD/SELL
X, y = create_binary_labels(df, config)
print(f"Labels created: Buy signals={int(np.sum(y))}, Hold/Sell={int(len(y)-np.sum(y))}, Positive rate={np.mean(y)*100:.1f}%")

# ============================================================================
# TRAIN/TEST SPLIT: Separate data into training and test sets
# ============================================================================
X_train, X_test, y_train, y_test = split_train_test(X, y, train_test_split)

# Normalize features using training set statistics (prevent data leakage)
X_train, X_test, xmin, xmax = normalize_features(X_train, X_test)
save_norm_params(xmin, xmax, config['norm_output'])

# Reshape labels to column vectors (required by model)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Convert to JAX arrays for GPU/memory-efficient computation
X_train = jnp.array(X_train)
y_train = jnp.array(y_train)
X_test = jnp.array(X_test)
y_test = jnp.array(y_test)

# ============================================================================
# MODEL INITIALIZATION: Create TKAN model parameters
# ============================================================================
input_dim = X_train.shape[-1]  # Number of features per time step
print(f"Train set: {X_train.shape}, Test set: {X_test.shape}, Input dimension: {input_dim}")

# Initialize random key for reproducibility
key = jax.random.key(seed)
key, k = jax.random.split(key)

# Create TKAN model with initialized parameters
params = init_tkan(input_dim, hidden_size, sub_dim, k)
total_params = sum(p.size for p in jax.tree_util.tree_leaves(params))
print(f"Model parameters: {total_params:,}")

# ============================================================================
# TRAINING: Optimize model with Adam optimizer
# ============================================================================
optimizer = optax.adam(learning_rate)
optimizer_state = optimizer.init(params)

start_time = time.time()

for epoch in range(epochs):
    key, subkey = jax.random.split(key)
    
    # Shuffle training data each epoch
    shuffled_indices = jax.random.permutation(subkey, len(X_train))
    
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    num_batches = 0
    
    # Mini-batch training
    for i in range(0, len(X_train), batch_size):
        # Get batch data
        batch_X = X_train[shuffled_indices[i:i+batch_size]]
        batch_y = y_train[shuffled_indices[i:i+batch_size]]
        
        # Compute loss and gradients
        def loss_fn(params):
            predictions = tkan_apply(params, batch_X, use_sigmoid=True)
            return binary_crossentropy(predictions, batch_y)

        loss, gradients = jax.value_and_grad(loss_fn)(params)
        epoch_loss += loss
        epoch_accuracy += compute_accuracy(tkan_apply(params, batch_X, use_sigmoid=True), batch_y)

        # Update parameters using Adam optimizer
        updates, optimizer_state = optimizer.update(gradients, optimizer_state)
        params = optax.apply_updates(params, updates)
        num_batches += 1
    
    # Report epoch progress
    avg_loss = epoch_loss / num_batches
    avg_accuracy = epoch_accuracy / num_batches
    print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, accuracy={avg_accuracy:.4f}")

# ============================================================================
# EVALUATION: Test model on held-out test set
# ============================================================================
test_predictions = tkan_apply(params, X_test, use_sigmoid=True)
test_accuracy = compute_accuracy(test_predictions, y_test)
elapsed_time = time.time() - start_time

print(f"Training completed in {elapsed_time:.1f}s, Test accuracy: {test_accuracy:.4f}")

# ============================================================================
# EXPORT: Save model to ONNX format for deployment
# ============================================================================
print("Exporting model to ONNX...")

result = to_onnx(
    make_apply_fn(params, use_sigmoid=True),
    inputs=[jax.ShapeDtypeStruct((1, sequence_length, input_dim), jnp.float32)],
    model_name='TKAN',
    return_mode='file',
    output_path=config['model_output']
)

print(f"Model saved to: {config['model_output']}")