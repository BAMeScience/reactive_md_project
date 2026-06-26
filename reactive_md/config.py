# reactive_md/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    data_file: str = "mixture_classical_after_300K_NPT.data"
    settings_file: str = "mixture.in.settings"

    # MD control
    dt: float = 1.0e-3
    steps: int = 2000
    check_every: int = 100
    max_events: int = 9999

    # Product-geometry preparation after an accepted reaction.
    # This is not a reaction gate; it only moves the leaving F away from P
    # before local relaxation of the product topology.
    r_pf_probe: float = 4.0 # controls how far the leaving F is moved away from P before relaxation.

    # Sigma-only reaction coordinate parameters.
    # sigma = d(P-F) - d(Li-F)
    # sigma_mid is the midpoint of the sigmoid reaction probability.
    # sigma_width controls how sharp the transition region is.
    sigma_mid: float = 0.0 # needed for rate mode
    sigma_width: float = 0.2 #needed for rate mode

    # Reaction mode and kinetics
    reaction_mode: str = "metropolis"
    reaction_rate_ps: float | None = None
    activation_energy_eV: float | None = None
    prefactor_ps: float | None = None
    max_reactions_per_check: int = 1

    # Thermodynamic / force-field settings
    temperature_k: float = 400.0
    kb_real: float = 0.0019872041

    r_cut: float = 15.0
    dr_threshold: float = 0.5

    # Atom types
    p_type: int = 6
    f_type: int = 7
    li_type: int = 8

    # Thermostat / random seed
    tau_T: float = 100.0
    prng_seed: int = 0

    # Output
    dump_file: str | None = None
    dump_every: int | None = None
    event_log_file: str | None = None

    candidate_log_file: str | None = None
    candidate_log_top_n: int = 10

    # Optional MACE energy for Metropolis acceptance
    use_mace_mc: bool = False
    mace_source: str = "mp"
    mace_variant: str | None = None
    mace_dr_threshold: float = 0.5
    mace_capacity_multiplier: float = 1.25

