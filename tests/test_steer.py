import jax
import jax.numpy as jnp

from locus import (
    apply_steering,
    difference_of_means,
    project_onto_atoms,
    property_dictionary,
    strength_scale,
)


def _cosine(a, b):
    a = jnp.asarray(a)
    b = jnp.asarray(b)
    return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b)))


def test_difference_of_means_cancels_shared_offset():
    d = 16
    n = 64
    property_dir = jax.random.normal(jax.random.PRNGKey(0), (d,))
    property_dir = property_dir / jnp.linalg.norm(property_dir)
    spurious = jax.random.normal(jax.random.PRNGKey(1), (d,))
    noise_pos = 0.01 * jax.random.normal(jax.random.PRNGKey(2), (n, d))
    noise_neg = 0.01 * jax.random.normal(jax.random.PRNGKey(3), (n, d))
    pos = property_dir[None, :] + noise_pos + spurious[None, :]
    neg = -property_dir[None, :] + noise_neg + spurious[None, :]
    direction = difference_of_means(pos, neg)
    assert _cosine(direction, property_dir) > 0.99


def test_project_onto_atoms_kills_spurious_component():
    property_dir = jnp.zeros((8,)).at[0].set(1.0)
    spurious_dir = jnp.zeros((8,)).at[1].set(1.7)
    direction = property_dir + spurious_dir
    projected = project_onto_atoms(direction, property_dir[:, None])
    assert abs(_cosine(projected, spurious_dir)) < 1e-5
    assert _cosine(projected, property_dir) > 0.99


def test_strength_scale_matches_baseline_l2():
    base = jnp.array([0.0, 4.0, 0.0])
    local = jnp.array([1.0, 0.0, 0.0])
    scale = strength_scale(base, local)
    assert abs(float(scale) - 4.0) < 1e-6
    zero = jnp.zeros_like(base)
    moved_local = apply_steering(zero, local, scale)
    moved_base = apply_steering(zero, base, 1.0)
    assert abs(float(jnp.linalg.norm(moved_local)) - float(jnp.linalg.norm(moved_base))) < 1e-5


def test_projected_steering_raises_only_property_logit():
    d = 8
    basis = jnp.eye(d)
    u_p = basis[:, 0]
    u_s = basis[:, 1]
    other = basis[:, 2:]
    W_U = jnp.concatenate([u_p[:, None], u_s[:, None], other], axis=1)
    r = jnp.zeros((d,))
    logits = W_U.T @ r
    s = u_p + u_s
    atoms = property_dictionary(W_U, [0])
    beta = 1.0
    r_proj = apply_steering(r, project_onto_atoms(s, atoms), beta)
    r_full = apply_steering(r, s, beta)
    logits_proj = W_U.T @ r_proj
    logits_full = W_U.T @ r_full
    assert float(logits_proj[0]) > float(logits[0])
    assert abs(float(logits_proj[1])) < 1e-5
    assert float(jnp.max(jnp.abs(logits_proj[1:]))) < 1e-5
    assert float(logits_full[0]) > float(logits[0])
    assert float(logits_full[1]) > float(logits[1])
