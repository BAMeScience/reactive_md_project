# Phase 5: source-grounded MC energy audit

**Primary source:** User-provided `rsmd-master.zip`, Git revision
`97ccc65dbb3d9d16e5e6a88f256d97a36c511740` (archive metadata),
source copyright 2020 Myra Biedermann.

## What the original code actually computes

1. `src/control/simulatorMetropolis.cpp`, `acceptance()` (lines 120-145):
   `energyDifference = energyParser->readPotentialEnergyDifference(...)`,
   then `energyDifference += candidate.getReactionEnergy()`, followed by
   `exp(-energyDifference/(R*T))`. The correction is supplied in the reaction
   template's `[energy]` directive (`src/parser/reactionParser.cpp`). There
   is no Metropolis-Hastings proposal-ratio factor.
2. `src/parser/energyParserGMX.cpp`, `readPotentialEnergyDifference()`
   (lines 29-46): subtracts pre-reaction potential from post-relaxation
   potential, optionally adding the difference in solvation interactions.
3. `src/engine/engineGMX.cpp`, `runEnergyComputation()` (lines 241-338):
   - `computeLocalPotentialEnergy=off`: full-system potential difference.
   - `computeLocalPotentialEnergy=on`, `computeSolvationPotentialEnergy=off`:
     energy difference of extracted reacting molecules only.
   - Both flags `on`: energy difference of extracted reacting molecules,
     **plus** LJ and Coulomb interaction energy differences with solvent.
   The solvation-only flag is invalid (`src/parameters/parameters.cpp`).
4. `src/parser/energyParserGMX.cpp`, `readPotentialEnergy()` supports either
   final snapshot (`averagePotentialEnergy=0`, default) or a time-averaged
   energy window. The Phase 5 implementation supports the default final
   snapshot only; averaging remains out of scope.
5. The paper's Section 3.2 states that including reactant-solvent interactions
   was essential to recover equilibrium product concentrations. The uploaded
   generic program source does not include the exact simulation input files,
   so it does **not** independently prove which flags were used in every
   published simulation.

## Change in the JAX-MD A2 implementation

For A2-only systems, the selected reacting pair has no bonded interactions
with the spectators. Consequently the source's local+solvation energy can be
computed as

    U_local+solvation(R) = U_full(R) - U_spectators(R_spectators)

at **both** the reactant and relaxed-product configurations. Subtracting the
spectator-only contribution removes changes in spectator-spectator energies
caused by Langevin relaxation, while retaining all A2 intrapair and A2-solvent
interactions. This identity assumes the A2 force field is pairwise additive
apart from bonded terms wholly within the reacting pair or spectators; it is
not claimed as a general formula for arbitrary many-body force fields.

`reactive_md/reactions/a2_energy.py` implements this identity by constructing
an independent spectator-only force field, preserving spectator topology and
nonbonded parameters. It explicitly rejects bonded terms crossing the reacting
pair / spectator boundary. The topology-dependent bond offset V0 is already
in `U_full`, so the difference includes V0 **exactly once**, matching the
separate `candidate.getReactionEnergy()` addition in the C++ source.

`reactive_md/reactions/a2_metropolis.py` and `a2_handlers.py` now default to
`energy_mode="local_solvation"`, matching the paper's stated methodological
preference. Set `energy_mode="full"` to recover the original program's
full-system option and the Phase 4 JAX-MD convention. Both modes use the
same one-candidate, relaxation-then-Metropolis workflow.

## Testing and limitations

- Added a real JAX-MD four-atom (2A+2B) regression verifying that the MC
  energy difference excludes B-B relaxation energy but includes A-B terms
  and V0. Existing two-atom MC tests continue to cover acceptance/rejection.
- Deterministic fake-force-field tests explicitly request `energy_mode="full"`
  because their minimal fakes do not implement spectator energy evaluation.
- The `local_solvation` mode rebuilds a spectator-only force field per trial;
  this is correct but may be expensive for large systems. Optimize only after
  verifying numerical parity and profiling.
- Exact parity with GROMACS energy-group rerun settings, sampling/averaging,
  finite precision, and the unpublished run inputs is not established.
- The model remains NVT; NPT/PEL II remain out of scope.
