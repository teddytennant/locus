"""Difference-of-means steering, subspace projection, and strength matching."""

import jax.numpy as jnp


def difference_of_means(pos, neg):
    """s = mean(pos, axis=0) - mean(neg, axis=0)."""
    return jnp.mean(jnp.asarray(pos), axis=0) - jnp.mean(jnp.asarray(neg), axis=0)


def project_onto_atoms(v, A):
    """Least-squares projection of v onto the columns of A, shape (d, k).

    Returns zeros when k is 0.
    """
    v = jnp.asarray(v)
    A = jnp.asarray(A)
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    if int(A.shape[-1]) == 0:
        return jnp.zeros_like(v)
    coef = jnp.linalg.lstsq(A, v)[0]
    return A @ coef


def _as_rows(vectors):
    if isinstance(vectors, (list, tuple)):
        if len(vectors) == 0:
            return jnp.zeros((0, 0))
        return jnp.stack([jnp.asarray(v) for v in vectors], axis=0)
    arr = jnp.asarray(vectors)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def strength_scale(base_vectors, local_vectors):
    """alpha = sum_i ||s_base_i||_2 / sum_i ||s_local_i||_2.

    Each argument is a list of residual-stream vectors or a stacked (n, d)
    array. alpha is 0 when the local mass is 0. A raw-head update should be
    mapped through W_O before this norm, ||W_O @ s_raw||_2.
    """
    base = _as_rows(base_vectors)
    local = _as_rows(local_vectors)
    base_sum = jnp.sum(jnp.linalg.norm(base, axis=-1))
    local_sum = jnp.sum(jnp.linalg.norm(local, axis=-1))
    if float(local_sum) == 0.0:
        return jnp.asarray(0.0, dtype=base_sum.dtype)
    return base_sum / local_sum


def apply_steering(activation, direction, beta):
    """steered = activation + beta * direction."""
    return jnp.asarray(activation) + beta * jnp.asarray(direction)
