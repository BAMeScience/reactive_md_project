from jax_md import units


def md_unit_factors(system: str):
    """Return mass conversion from amu and kB in MD energy units/K.

    Integrator convention:
      position: Angstrom
      time: ps
      energy: kcal/mol (real) or eV (metal)
    """
    if system == "real":
        unit = units.real_unit_system()
        ps_in_jax_time = 1000.0 * float(unit["time"])
    elif system == "metal":
        unit = units.metal_unit_system()
        ps_in_jax_time = float(unit["time"])
    else:
        raise ValueError(f"Unsupported MD unit system: {system}")

    mass_factor = float(unit["mass"]) / ps_in_jax_time**2
    kb_md = float(unit["temperature"])

    return mass_factor, kb_md
