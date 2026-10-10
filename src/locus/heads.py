"""RMSNorm, head-level unembedding, and EVR head selection."""

import jax.numpy as jnp


def rms_norm(o, gamma=None, eps=1e-6):
    """o_bar = o / sqrt(mean(o^2) + eps) * gamma. gamma defaults to ones."""
    o = jnp.asarray(o)
    if gamma is None:
        gamma = jnp.ones((o.shape[-1],), dtype=o.dtype)
    else:
        gamma = jnp.asarray(gamma)
    rms = jnp.sqrt(jnp.mean(jnp.square(o), axis=-1, keepdims=True) + eps)
    return o / rms * gamma


def _population_threshold(evrs):
    values = jnp.asarray(evrs, dtype=jnp.float32).reshape(-1)
    mean = jnp.mean(values)
    # Population std: divide by n, not n - 1.
    std = jnp.sqrt(jnp.mean(jnp.square(values - mean)))
    return values, mean + 2.0 * std


def select_heads(evrs):
    """Indices with e > mean(e) + 2 * population_std(e).

    A zero std makes the threshold equal the mean, so the strict inequality
    selects nothing. evrs may be any shape; selection is on the flat vector.
    """
    values, thresh = _population_threshold(evrs)
    return jnp.where(values > thresh)[0]


def select_heads_matrix(evrs_2d):
    """Same rule as select_heads, returned as (layer, head) pairs."""
    grid = jnp.asarray(evrs_2d)
    if grid.ndim != 2:
        raise ValueError("evrs_2d must have shape (layers, heads)")
    n_heads = int(grid.shape[1])
    pairs = []
    for idx in select_heads(grid).tolist():
        idx = int(idx)
        pairs.append((idx // n_heads, idx % n_heads))
    return pairs


def head_unembedding(U_P, W_O, gamma, rms_hat):
    """U_e = U_P.T @ diag(gamma / rms_hat) @ W_O, shape (|P|, d_h).

    Rows are property directions in raw-head space. For SOMP there, pass U_e.T
    so atoms are columns.
    """
    U_P = jnp.asarray(U_P)
    W_O = jnp.asarray(W_O)
    scale = jnp.asarray(gamma) / rms_hat
    return U_P.T @ (scale[:, None] * W_O)
