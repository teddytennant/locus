"""Property dictionary and simultaneous orthogonal matching pursuit."""

import jax.numpy as jnp


def property_dictionary(W_U, token_ids):
    """Column subset U_P = W_U[:, P], shape (d, |P|)."""
    W_U = jnp.asarray(W_U)
    ids = jnp.asarray(token_ids, dtype=jnp.int32)
    return W_U[:, ids]


def somp(U_P, Y, n):
    """Simultaneous OMP for min ||Y - U_P C||_F^2 with at most n nonzero rows.

    U_P is (d, m), Y is (d, N). Ties take the smallest atom index. Stops early
    if the best remaining score is 0, or when fewer than n atoms remain.
    Returns the support as a list of ints and a dense C of shape (m, N).
    """
    U_P = jnp.asarray(U_P)
    Y = jnp.asarray(Y)
    m = int(U_P.shape[1])
    n_cols = int(Y.shape[1])
    n_iters = min(int(n), m)
    R = Y
    support = []
    C_s = None

    for _ in range(n_iters):
        corr = U_P.T @ R
        scores = jnp.sum(jnp.square(corr), axis=1)
        best = None
        best_score = 0.0
        used = set(support)
        for j in range(m):
            if j in used:
                continue
            score = float(scores[j])
            # Strict greater-than keeps the smallest index on a tie.
            if best is None or score > best_score:
                best = j
                best_score = score
        if best is None or best_score <= 0.0:
            break
        support.append(best)
        U_s = U_P[:, jnp.asarray(support, dtype=jnp.int32)]
        C_s = jnp.linalg.lstsq(U_s, Y)[0]
        R = Y - U_s @ C_s

    C = jnp.zeros((m, n_cols), dtype=Y.dtype)
    if support:
        C = C.at[jnp.asarray(support, dtype=jnp.int32)].set(C_s)
    return support, C


def evr(Y, U_P, C):
    """Explained-variance ratio 1 - ||Y - U_P C||_F^2 / ||Y||_F^2.

    Returns 1 when ||Y||_F is 0.
    """
    Y = jnp.asarray(Y)
    resid = Y - jnp.asarray(U_P) @ jnp.asarray(C)
    denom = jnp.sum(jnp.square(Y))
    if float(denom) == 0.0:
        return jnp.asarray(1.0, dtype=Y.dtype)
    return 1.0 - jnp.sum(jnp.square(resid)) / denom
