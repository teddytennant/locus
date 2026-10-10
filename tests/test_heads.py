import jax
import jax.numpy as jnp

from locus import head_unembedding, select_heads, select_heads_matrix


def _threshold(evrs):
    values = jnp.asarray(evrs, dtype=jnp.float32)
    mean = jnp.mean(values)
    std = jnp.sqrt(jnp.mean(jnp.square(values - mean)))
    return float(mean + 2.0 * std)


def test_select_heads_threshold():
    evrs = jnp.array([0.1, 0.1, 0.1, 0.9])
    thresh = _threshold(evrs)
    selected = [int(i) for i in select_heads(evrs).tolist()]
    expected = [i for i, value in enumerate(evrs.tolist()) if value > thresh]
    assert selected == expected
    # A lone outlier in a sample of 4 has z-score sqrt(3) < 2, so mean+2*std
    # does not clear 0.9. The same rule isolates 0.9 once n is large enough.
    assert evrs[3] <= thresh
    assert selected == []

    wide = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    wide_thresh = _threshold(wide)
    wide_selected = [int(i) for i in select_heads(wide).tolist()]
    assert wide_selected == [i for i, value in enumerate(wide.tolist()) if value > wide_thresh]
    assert wide_selected == [5]
    assert float(wide[5]) > wide_thresh


def test_select_heads_all_equal_selects_nothing():
    evrs = jnp.array([0.4, 0.4, 0.4, 0.4])
    assert select_heads(evrs).size == 0
    assert _threshold(evrs) == float(jnp.mean(evrs))


def test_select_heads_matrix_uses_flat_rule():
    grid = jnp.array(
        [
            [0.1, 0.1, 0.1],
            [0.1, 0.1, 0.9],
        ]
    )
    assert select_heads_matrix(grid) == [(1, 2)]


def test_head_unembedding_matches_diag_formula():
    U_P = jax.random.normal(jax.random.PRNGKey(0), (4, 3))
    W_O = jax.random.normal(jax.random.PRNGKey(1), (4, 2))
    gamma = jnp.ones((4,))
    rms_hat = 2.0
    got = head_unembedding(U_P, W_O, gamma, rms_hat)
    expected = U_P.T @ jnp.diag(gamma / rms_hat) @ W_O
    assert got.shape == (3, 2)
    assert jnp.allclose(got, expected, atol=1e-6)
