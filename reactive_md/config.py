# reactive_md/config.py
from dataclasses import dataclass

@dataclass(frozen=True)
class SimConfig:
    data_file: str = "mixture_classical_after_300K_NPT.data"
    settings_file: str = "mixture.in.settings"

    dt: float = 1.0e-3
    steps: int = 2000
    check_every: int = 50
    max_events: int = 9999

    r_on: float = 4.0
    temperature_k: float = 400.0
    kb_real: float = 0.0019872041  # kcal/mol/K in LAMMPS "real" units

    r_cut: float = 15.0
    dr_threshold: float = 0.5

    p_type: int = 6
    f_type: int = 7
    li_type: int = 8

    tau_T: float = 100.0  # Nose-Hoover tau
    prng_seed: int = 0

    # Logging
    log_file: str = "reactive_md.log"
    log_to_console: bool = True
    log_level: str = "INFO"   # DEBUG / INFO / WARNING / ERROR

