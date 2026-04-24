import os
os.environ['JAX_PLATFORMS'] = 'cpu'

import pytest
import numpy as np
import jax
import jax.numpy as jnp
from tkan.tkan_model import init_tkan, tkan_apply, binary_crossentropy, compute_accuracy
from tkan.labels import get_trade_label, normalize_features, split_train_test


def test_tkan_shape():
    key = jax.random.key(0)
    params = init_tkan(input_dim=4, hidden=8, sub=4, key=key)
    x = jnp.ones((2, 10, 4))
    out = tkan_apply(params, x, use_sigmoid=True)
    assert out.shape == (2, 1)


def test_tkan_output_range():
    key = jax.random.key(1)
    params = init_tkan(input_dim=4, hidden=8, sub=4, key=key)
    x = jax.random.normal(key, (4, 10, 4))
    out = tkan_apply(params, x, use_sigmoid=True)
    assert jnp.all(out >= 0) and jnp.all(out <= 1)


def test_tkan_unique_keys():
    key = jax.random.key(42)
    k = jax.random.split(key, 7)
    for i in range(7):
        for j in range(i + 1, 7):
            assert not jnp.array_equal(k[i], k[j])


def test_trade_label_tp():
    assert get_trade_label(100, [102, 101], 1.5, 2.0) == 1


def test_trade_label_sl():
    assert get_trade_label(100, [97, 102], 1.5, 2.0) == 0


def test_trade_label_no_hit():
    assert get_trade_label(100, [100.5, 99.5], 1.5, 2.0) == 0


def test_normalize_per_feature():
    X_tr = np.array([[[1, 10], [2, 20]], [[3, 30], [4, 40]]], dtype=np.float32)
    X_te = np.array([[[2, 20], [3, 30]]], dtype=np.float32)
    Xn_tr, Xn_te, xmin, xmax = normalize_features(X_tr, X_te)
    assert Xn_tr.min() >= 0.0 and Xn_tr.max() <= 1.0
    assert xmin.shape == (2,)
    assert xmax.shape == (2,)


def test_split():
    X = np.ones((100, 5, 4), dtype=np.float32)
    y = np.zeros(100, dtype=np.float32)
    X_tr, X_te, y_tr, y_te = split_train_test(X, y, 0.8)
    assert len(X_tr) == 80 and len(X_te) == 20


def test_loss_finite():
    key = jax.random.key(7)
    params = init_tkan(input_dim=4, hidden=8, sub=4, key=key)
    x = jax.random.normal(key, (8, 10, 4))
    pred = tkan_apply(params, x, use_sigmoid=True)
    target = jnp.ones((8, 1))
    loss = binary_crossentropy(pred, target)
    assert jnp.isfinite(loss)
