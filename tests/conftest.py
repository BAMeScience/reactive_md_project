"""Module-level test categories; no tests are skipped or deselected here."""

import pytest


_A2_REFERENCE_MODULES = {
    "test_a2_energy_reference_phase7.py",
    "test_a2_md_only_reference.py",
    "test_a2_mh_exact_reference.py",
    "test_a2_mh_kernel_reference.py",
    "test_a2_mh_mixing_reference.py",
    "test_a2_mh_moving_reference.py",
    "test_a2_mh_sampling_reference.py",
    "test_a2_thermostat_calibration_math.py",
}


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items):
    for item in items:
        name = item.path.name
        if name.startswith("test_a2_") or name == "test_md_a2_driver.py":
            item.add_marker(pytest.mark.a2)
        if name in _A2_REFERENCE_MODULES:
            item.add_marker(pytest.mark.a2_reference)
        if name.startswith("test_mace_"):
            item.add_marker(pytest.mark.mace_interface)
