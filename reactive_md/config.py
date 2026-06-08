# reactive_md/config.py
from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    data_file: str = "mixture_classical_after_300K_NPT.data"
    settings_file: str = "mixture.in.settings"

    dt: float = 1.0e-3
    steps: int = 2000
    check_every: int = 100
    max_events: int = 9999

    r_lif_on: float = 2.4
    r_pf_break: float = 1.65
    r_pf_probe: float = 4.0

    rate_pf_mode: str = "sigmoid"
    rate_pf_mid: float = 1.62
    rate_pf_width: float = 0.02

    reaction_mode: str = "metropolis"
    reaction_rate_ps: float | None = None
    activation_energy_eV: float | None = None
    prefactor_ps: float | None = None
    max_reactions_per_check: int = 1

    temperature_k: float = 400.0
    kb_real: float = 0.0019872041

    r_cut: float = 15.0
    dr_threshold: float = 0.5

    p_type: int = 6
    f_type: int = 7
    li_type: int = 8

    tau_T: float = 100.0
    prng_seed: int = 0

    dump_file: str | None = None
    dump_every: int | None = None
    event_log_file: str | None = None

    candidate_log_file: str | None = None
    candidate_log_top_n: int = 10

    use_mace_mc: bool = False
    mace_source: str = "mp"
    mace_variant: str | None = None
    mace_dr_threshold: float = 0.5
    mace_capacity_multiplier: float = 1.25
