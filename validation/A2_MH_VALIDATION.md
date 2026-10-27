# A₂ fixed-coordinate Metropolis–Hastings validation

**Reference:** user-supplied `reactive_md_project-develop.zip` (26 October 2026).
This validation is additive: **no production Python module was modified**.
In particular, the previously validated rate-based A₂ chemistry is untouched.

## Model being tested

The `A2MetropolisHandler` calls `a2_metropolis_reactive_cycle` to attempt **one**
A + A ⇌ A₂ topology move at the **same atomic positions and velocities**.
An association is eligible only when two free A atoms are within the
association cutoff. Any existing A₂ bond can be proposed for dissociation,
but the algorithm rejects a dissociation if the reverse association would
be outside the cutoff at the unchanged coordinates.

Let `s` denote the bond matching, `R` the fixed coordinates, and `U_s(R)`
the **total** force-field energy, including one `bond_energy_offset` for each
A₂ dimer. At fixed `R` the intended conditional target is

    pi(s | R) ∝ exp[-U_s(R) / (k_B T)].

No separate equilibrium constant, standard-state factor, or chemical
potential is currently included. Consequently this is **not** a calibrated
chemical equilibrium model for any particular experimental reaction.

With `p` the probability of choosing the association direction, and
`N_a(s)` and `N_d(s)` the eligible candidate counts:

    q(s → t) = p / N_a(s)        for association
    q(s → t) = (1-p) / N_d(s)    for dissociation

The direction is chosen **before** checking candidate availability; a
missing direction is a self transition and its probability is not
redistributed. For a reachable reverse transition:

    alpha(s → t) = min(1, exp[-(U_t-U_s)/(k_B T)] * q(t→s)/q(s→t)).

This ratio is essential: eligible candidate counts generally differ between
states. `alpha` must be zero when the reverse proposal has zero probability.

## New tests

| File | Scope | Where it can run |
| --- | --- | --- |
| `tests/test_a2_mh_exact_reference.py` | Independent 10-state/12-edge finite-state kernel; detailed balance, row sums, stationary Boltzmann distribution, null directions, several `p` and `kT` values | Any Python with pytest |
| `tests/test_a2_metropolis_detailed_balance.py` | Actual production MH implementation on all **24 directed edges** with a synthetic topology-energy evaluator; proposal ratios, acceptance, state changes, impossible reverse, invalid inputs, null direction | HPC JAX/JAX-MD environment |
| `tests/test_a2_metropolis_detailed_balance.py` | Actual JAX-MD energy and exclusion-mask construction, per-bond offset, forward/reverse energy and MH acceptance round trip | HPC JAX/JAX-MD environment |

The synthetic evaluator is intentional: it isolates the *MH algorithm* from
JAX-MD and compares it to an independent exact finite-state calculation. The
separate integration test uses real JAX-MD energies to check the coupling.

### Commands

From the repository root, in the **same HPC environment** that passed the
previous A₂ kinetic regression:

```bash
python -m pytest -q tests/test_a2_mh_exact_reference.py
python -m pytest -q tests/test_a2_metropolis_detailed_balance.py
python -m pytest -q tests/
```

The authoring environment has pytest and JAX but **does not have `jax_md`**.
The independent 10-state test passed locally (**10 passed**). The production
MH and real force-field tests are **not yet executed**; do not report them as
passing until the HPC test output is available.

## Scientific limitations / follow-up

1. **Starting with pre-existing dimers:** the initial `ff` passed into the
   A₂ MH driver must already include `n_dimers * bond_energy_offset` in its
   total energy. Trial force fields do include this offset, but a manually
   initialized `build_forcefield(...)` without `energy_offset=` does not.
   Starting with all monomers needs a zero offset and avoids this mismatch.
   A future production safeguard should validate or reconstruct the initial
   force field when the starting topology contains A₂ bonds.
2. **Restricted reachability:** at a fixed `R`, topology states connected
   only by bonds longer than the association cutoff are not mutually
   reachable. Detailed balance is checked for reachable edges; this does
   **not** establish global ergodicity of the combined MD/MC simulation.
3. **Finite-state detailed balance is necessary, not sufficient:** to claim
   the coupled MD/MC algorithm samples a canonical joint distribution,
   independently validate the NVT integrator, state handoff, energy/force
   synchronization, temperature, and long-run population statistics.
4. **No physical standard-state correction:** the dimer offset changes the
   relative Boltzmann weights but does not by itself define a macroscopic
   equilibrium constant. Any physical interpretation requires an explicit
   thermodynamic convention and concentration/volume analysis.
5. **No kinetic inference:** MC attempt frequencies and accepted moves are
   not physical association/dissociation rate constants. Do not compare MH
   event counts directly with the validated rate-based 10 ns trajectory.
6. **Precision:** the real-force-field test uses tolerances appropriate for
   float32 JAX. Any non-finite energy or overflow must be investigated,
   not masked by widening tolerances indiscriminately.

## Gate before declaring MH validated

- [x] Independent 10-state reference calculation passes locally.
- [ ] Production MH implementation passes all directed-edge tests on HPC.
- [ ] Real JAX-MD energy/offset and forward/reverse round-trip test passes.
- [ ] Full original suite remains green on the same HPC environment.
- [ ] Initial-bond offset policy is verified for the intended initial state.
- [ ] Coupled NVT/MC equilibrium and sampling checks are completed.

Only after these gates can we make a scientifically scoped equilibrium claim.
