import jax
import jax.numpy as jnp

from locus import evr, somp


def test_somp_recovers_planted_support():
    raw = jax.random.normal(jax.random.PRNGKey(0), (16, 8))
    U, _ = jnp.linalg.qr(raw)
    coeffs = jnp.array(
        [
            [1.2, -0.4, 0.7, 0.3, -1.1],
            [0.5, 0.9, -0.2, 1.4, 0.6],
        ]
    )
    Y = U[:, jnp.array([1, 3])] @ coeffs
    support, C = somp(U, Y, n=2)
    assert set(support) == {1, 3}
    assert C.shape == (8, 5)
    off = [j for j in range(8) if j not in support]
    assert float(jnp.max(jnp.abs(C[jnp.array(off)]))) == 0.0
    assert float(evr(Y, U, C)) > 0.999


def test_somp_orthogonal_target_has_low_evr():
    raw = jax.random.normal(jax.random.PRNGKey(0), (16, 8))
    U, _ = jnp.linalg.qr(raw)
    Y_raw = jax.random.normal(jax.random.PRNGKey(1), (16, 5))
    Y = Y_raw - U @ (U.T @ Y_raw)
    _support, C = somp(U, Y, n=2)
    assert float(evr(Y, U, C)) < 0.05


def test_somp_tie_break_picks_smallest_index():
    atom = jnp.array([0.0, 1.0, 0.0])
    other = jnp.array([1.0, 0.0, 0.0])
    # Columns 1 and 2 are identical, so their scores tie. Column 0 does not.
    U = jnp.stack([other, atom, atom], axis=1)
    Y = atom[:, None]
    support, _C = somp(U, Y, n=2)
    assert support[0] == 1
