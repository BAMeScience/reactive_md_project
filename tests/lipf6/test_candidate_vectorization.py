"""Equivalence of batched LiPF6 forward candidate geometry to scalar search."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from reactive_md.reactions.lipf6 import ReactionCandidate, find_reaction_candidates


def scalar_reference(positions, pf6, lithium, reacted, displacement):
    candidates = []
    for k, row in enumerate(pf6):
        if reacted[k]:
            continue
        for li in lithium:
            for f in row[1:]:
                d_lif = float(np.linalg.norm(np.asarray(
                    displacement(positions[li], positions[f])
                )))
                d_pf = float(np.linalg.norm(np.asarray(
                    displacement(positions[row[0]], positions[f])
                )))
                candidates.append(ReactionCandidate(k, int(li), int(f), d_lif, d_pf))
    return sorted(candidates, key=lambda c: c.d_pf - c.d_lif, reverse=True)


def identity(c):
    return c.k_pf6, c.li_idx, c.leave_F, c.direction


@pytest.mark.parametrize('periodic', [False, True])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_forward_matches_scalar_reference(periodic, dtype):
    if dtype == np.float64 and not jax.config.jax_enable_x64:
        pytest.skip('Run with JAX_ENABLE_X64=True to test float64')
    rng = np.random.default_rng(751)
    positions = jnp.asarray(rng.uniform(0, 12, (25, 3)), dtype=dtype)
    assert positions.dtype == dtype
    pf6 = np.arange(21, dtype=np.int32).reshape(3, 7)
    lithium = np.array([24, 22, 21, 23], dtype=np.int32)
    reacted = np.array([False, True, False])

    def displacement(a, b):
        delta = a - b
        return delta - 12 * jnp.round(delta / 12) if periodic else delta

    expected = scalar_reference(positions, pf6, lithium, reacted, displacement)
    actual = find_reaction_candidates(
        positions, pf6, lithium, displacement, pf6_reacted_np=reacted
    )
    assert len(actual) == 48
    assert [identity(c) for c in actual] == [identity(c) for c in expected]
    np.testing.assert_allclose(
        [(c.d_lif, c.d_pf) for c in actual],
        [(c.d_lif, c.d_pf) for c in expected],
        rtol=0, atol=1e-6 if dtype == np.float32 else 1e-12,
    )


def test_exact_ties_preserve_pf6_li_f_order():
    positions = jnp.zeros((17, 3))
    pf6 = np.arange(14).reshape(2, 7)
    lithium = np.array([16, 14, 15])
    reacted = np.array([False, False])
    actual = find_reaction_candidates(
        positions, pf6, lithium, lambda a, b: a - b, pf6_reacted_np=reacted
    )
    assert [identity(c) for c in actual] == [
        (k, li, f, 'forward') for k, row in enumerate(pf6)
        for li in lithium for f in row[1:]
    ]


@pytest.mark.parametrize('case', ['all_reacted', 'no_pf6', 'no_li'])
def test_empty_candidates_do_not_call_displacement(case):
    pf6 = np.arange(7).reshape(1, 7) if case != 'no_pf6' else np.empty((0, 7), int)
    lithium = np.array([7]) if case != 'no_li' else np.empty(0, int)
    reacted = np.full(len(pf6), case == 'all_reacted')

    def displacement(a, b):
        pytest.fail('Empty search must not evaluate displacement')

    assert find_reaction_candidates(
        jnp.zeros((8, 3)), pf6, lithium, displacement, pf6_reacted_np=reacted
    ) == []


def test_host_only_displacement_remains_supported():
    positions = jnp.arange(24, dtype=jnp.float32).reshape(8, 3)
    pf6 = np.arange(7).reshape(1, 7)
    lithium = np.array([7])
    reacted = np.array([False])

    def displacement(a, b):
        return np.asarray(a) - np.asarray(b)

    assert find_reaction_candidates(
        positions, pf6, lithium, displacement, pf6_reacted_np=reacted
    ) == scalar_reference(positions, pf6, lithium, reacted, displacement)
