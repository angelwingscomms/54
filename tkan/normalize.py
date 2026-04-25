import jax
import jax.numpy as jnp


def normalize(xmin, xmax, X_tr, X_te, y_tr, y_te):
    X_tr = (X_tr - xmin) / (xmax - xmin + 1e-8)
    X_te = (X_te - xmin) / (xmax - xmin + 1e-8)
    return X_tr, X_te, y_tr, y_te