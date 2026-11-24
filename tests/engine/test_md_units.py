import pytest

from reactive_md.md_units import md_unit_factors


@pytest.mark.parametrize("system, mass_expected, kb_expected", [
    ("real", 1.0 / 418.4, 0.0019872041),
    ("metal", 1.0 / 9648.5332, 8.617333262e-5),
])
def test_md_units_physical_constants(system, mass_expected, kb_expected):
    mass, kb = md_unit_factors(system)
    assert mass == pytest.approx(mass_expected, rel=1e-4)
    assert kb == pytest.approx(kb_expected, rel=1e-4)


def test_md_units_energy_conversion_consistency():
    mass_real, kb_real = md_unit_factors("real")
    mass_metal, kb_metal = md_unit_factors("metal")
    assert mass_real / mass_metal == pytest.approx(23.0605489, rel=1e-4)
    assert kb_real / kb_metal == pytest.approx(23.0605489, rel=1e-4)


def test_md_units_reject_unknown_system():
    with pytest.raises(ValueError, match="Unsupported MD unit system"):
        md_unit_factors("unknown")
