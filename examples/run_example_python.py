import os
os.environ['KERAS_BACKEND'] = 'jax'

import numpy as np
import pandas as pd
import time

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, GRU
from keras.optimizers import Adam

from tkan import TKAN

import warnings
warnings.filterwarnings('ignore')

keras.utils.set_random_seed(1)

N_MAX_EPOCHS = 10
BATCH_SIZE = 128

early_stopping_callback = lambda: keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.00001,
    patience=5,
    mode="min",
    restore_best_weights=True,
    start_from_epoch=3,
)
lr_callback = lambda: keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=3,
    mode="min",
    min_delta=0.00001,
    min_lr=0.00005,
    verbose=0,
)
callbacks = lambda: [early_stopping_callback(), lr_callback()]


class MinMaxScaler:
    def __init__(self, feature_axis=None, minmax_range=(0, 1)):
        self.feature_axis = feature_axis
        self.min_ = None
        self.max_ = None
        self.scale_ = None
        self.minmax_range = minmax_range

    def fit(self, X):
        if X.ndim == 3 and self.feature_axis is not None:
            axis = tuple(i for i in range(X.ndim) if i != self.feature_axis)
            self.min_ = np.min(X, axis=axis)
            self.max_ = np.max(X, axis=axis)
        elif X.ndim == 2:
            self.min_ = np.min(X, axis=0)
            self.max_ = np.max(X, axis=0)
        elif X.ndim == 1:
            self.min_ = np.min(X)
            self.max_ = np.max(X)
        else:
            raise ValueError("Data must be 1D, 2D, or 3D.")
        self.scale_ = self.max_ - self.min_
        return self

    def transform(self, X):
        X_scaled = (X - self.min_) / self.scale_
        X_scaled = X_scaled * (self.minmax_range[1] - self.minmax_range[0]) + self.minmax_range[0]
        return X_scaled

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        X = (X_scaled - self.minmax_range[0]) / (self.minmax_range[1] - self.minmax_range[0])
        X = X * self.scale_ + self.min_
        return X


def generate_data(df, sequence_length, n_ahead=1):
    scaler_df = df.copy().shift(n_ahead).rolling(24 * 14).median()
    tmp_df = df.copy() / scaler_df
    tmp_df = tmp_df.iloc[24 * 14 + n_ahead:].fillna(0.)
    scaler_df = scaler_df.iloc[24 * 14 + n_ahead:].fillna(0.)

    def prepare_sequences(df, scaler_df, n_history, n_future):
        X, y, y_scaler = [], [], []
        num_features = df.shape[1]
        
        for i in range(n_history, len(df) - n_future + 1):
            X.append(df.iloc[i - n_history:i].values)
            y.append(df.iloc[i:i + n_future, 0:1].values)
            y_scaler.append(scaler_df.iloc[i:i + n_future, 0:1].values)
        
        X, y, y_scaler = np.array(X), np.array(y), np.array(y_scaler)
        return X, y, y_scaler
    
    X, y, y_scaler = prepare_sequences(tmp_df, scaler_df, sequence_length, n_ahead)
    
    train_test_separation = int(len(X) * 0.8)
    X_train_unscaled, X_test_unscaled = X[:train_test_separation], X[train_test_separation:]
    y_train_unscaled, y_test_unscaled = y[:train_test_separation], y[train_test_separation:]
    
    X_scaler = MinMaxScaler(feature_axis=2)
    X_train = X_scaler.fit_transform(X_train_unscaled)
    X_test = X_scaler.transform(X_test_unscaled)
    
    y_scaler = MinMaxScaler(feature_axis=2)
    y_train = y_scaler.fit_transform(y_train_unscaled)
    y_test = y_scaler.transform(y_test_unscaled)
    
    y_train = y_train.reshape(y_train.shape[0], -1)
    y_test = y_test.reshape(y_test.shape[0], -1)
    
    return X_scaler, X_train, X_test, X_train_unscaled, X_test_unscaled, y_scaler, y_train, y_test, y_train_unscaled, y_test_unscaled


print("Loading data...")
df = pd.read_parquet('examples/data.parquet')
df = df[(df.index >= pd.Timestamp('2020-01-01')) & (df.index < pd.Timestamp('2023-01-01'))]
assets = ['BTC', 'ETH', 'ADA', 'XMR', 'EOS', 'MATIC', 'TRX', 'FTM', 'BNB', 'XLM', 'ENJ', 'CHZ', 'BUSD', 'ATOM', 'LINK', 'ETC', 'XRP', 'BCH', 'LTC']
df = df[[c for c in df.columns if 'quote asset volume' in c and any(asset in c for asset in assets)]]
df.columns = [c.replace(' quote asset volume', '') for c in df.columns]

print(f"Data shape: {df.shape}")
print(f"Assets: {list(df.columns)}")

print("\nGenerating sequences...")
sequence_length = 45
n_ahead = 1
X_scaler, X_train, X_test, X_train_unscaled, X_test_unscaled, y_scaler, y_train, y_test, y_train_unscaled, y_test_unscaled = generate_data(df, sequence_length, n_ahead)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

print("\nCreating TKAN model...")
model = Sequential([
    Input(shape=X_train.shape[1:]),
    TKAN(100, return_sequences=True),
    TKAN(100, sub_kan_output_dim=20, sub_kan_input_dim=20, return_sequences=False),
    Dense(units=n_ahead, activation='linear')
], name='TKAN')

optimizer = Adam(0.001)
model.compile(optimizer=optimizer, loss='mean_squared_error')

model.summary()

print("\nTraining TKAN model...")
start_time = time.time()
history = model.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=N_MAX_EPOCHS,
    validation_split=0.2,
    callbacks=callbacks(),
    shuffle=True,
    verbose=1
)
tkan_time = time.time() - start_time

preds = model.predict(X_test, verbose=0)
mse = np.mean((y_test - preds) ** 2)
rmse = np.sqrt(mse)

print(f"\nTKAN Training time: {tkan_time:.2f}s")
print(f"TKAN MSE: {mse:.6f}")
print(f"TKAN RMSE: {rmse:.6f}")

print("\nCreating GRU model for comparison...")
model_gru = Sequential([
    Input(shape=X_train.shape[1:]),
    GRU(100, return_sequences=True),
    GRU(100, return_sequences=False),
    Dense(units=n_ahead, activation='linear')
], name='GRU')

optimizer_gru = Adam(0.001)
model_gru.compile(optimizer=optimizer_gru, loss='mean_squared_error')

print("\nTraining GRU model...")
start_time = time.time()
history_gru = model_gru.fit(
    X_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=N_MAX_EPOCHS,
    validation_split=0.2,
    callbacks=callbacks(),
    shuffle=True,
    verbose=1
)
gru_time = time.time() - start_time

preds_gru = model_gru.predict(X_test, verbose=0)
mse_gru = np.mean((y_test - preds_gru) ** 2)
rmse_gru = np.sqrt(mse_gru)

print(f"\nGRU Training time: {gru_time:.2f}s")
print(f"GRU MSE: {mse_gru:.6f}")
print(f"GRU RMSE: {rmse_gru:.6f}")


print("\n" + "="*50)
print("RESULTS SUMMARY")
print("="*50)
print(f"TKAN  - Time: {tkan_time:.2f}s, RMSE: {rmse:.6f}")
print(f"GRU   - Time: {gru_time:.2f}s, RMSE: {rmse_gru:.6f}")
print("="*50)