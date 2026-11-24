import numpy as np
import jax.numpy as jnp
import types
import sys
import pytest

from reactive_md.md_units import md_unit_factors

from reactive_md.config import SimConfig
import reactive_md.main as main_mod
from reactive_md.apps import lipf6 as lipf6_app

@pytest.fixture
def mocked_parser(monkeypatch):
    # Provide a minimal parse_lammps_data output matching your contract
    def fake_parse_lammps_data(data_file, settings_file):
        positions = np.array([[0,0,0],[1,0,0]], dtype=np.float32)
        # empty bonded terms
        bonds = (np.zeros((0,2), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.float32))
        angles = (np.zeros((0,3), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.float32))
        torsions = (np.zeros((0,4), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        impropers = (np.zeros((0,4), np.int32), np.zeros((0,), np.float32), np.zeros((0,), np.int32), np.zeros((0,), np.float32))
        # nonbonded
        charges = np.array([0.0, 0.0], dtype=np.float32)
        sigmas = np.array([3.0, 3.0], dtype=np.float32)
        eps = np.array([0.1, 0.1], dtype=np.float32)
        nonbonded = (charges, sigmas, eps, None, None)

        molecule_id = np.array([1,1], dtype=np.int32)
        box = jnp.array([10.0,10.0,10.0], dtype=jnp.float32)
        masses = np.array([1.0, 1.0], dtype=np.float32)
        atom_types = np.array([1,1], dtype=np.int32)
        return positions, bonds, angles, torsions, impropers, nonbonded, molecule_id, box, masses, atom_types

    monkeypatch.setattr(lipf6_app, "parse_lammps_data", fake_parse_lammps_data)


def test_main_smoke_with_mocked_parser(mocked_parser):
    # prevent MD from running long: make steps very small
    cfg = SimConfig(steps=2, check_every=1, max_events=0)
    # Should not raise
    main_mod.main(cfg)



@pytest.mark.parametrize("unit_system", ["real", "metal"])
@pytest.mark.parametrize("mode", ["metropolis", "rate"])
def test_mace_relaxation_uses_md_thermal_units(monkeypatch, mocked_parser,
                                              unit_system, mode):
    # No optional MACE dependencies or model weights are required.
    neighbor = object()
    setup = types.ModuleType("reactive_md.mace_setup")
    setup.build_mace_model = lambda **kw: (None, None, None)
    energy = types.ModuleType("reactive_md.mace_energy")
    energy.build_mace_energy_system = lambda **kw: (
        None, None, neighbor, lambda nl: lambda R, **kw: jnp.asarray(2.0)
    )
    monkeypatch.setitem(sys.modules, setup.__name__, setup)
    monkeypatch.setitem(sys.modules, energy.__name__, energy)
    captured = {}

    def reaction_stub(*args, **kwargs):
        captured.update(kwargs)

    def driver_stub(key, **kwargs):
        R = kwargs["init_positions"]
        kwargs["reaction_step_fn"](
            key, R, kwargs["ff"], kwargs["sys"], jnp.zeros_like(R)
        )
        return kwargs

    monkeypatch.setattr(lipf6_app, "run_md_nvt_with_reactions", driver_stub)
    monkeypatch.setattr(lipf6_app, "maybe_react_one_event", reaction_stub)
    monkeypatch.setattr(lipf6_app, "maybe_react_rate_events", reaction_stub)
    cfg = SimConfig(md_unit_system=unit_system, reaction_mode=mode,
                    use_mace_md=True, use_mace_mc=(mode == "metropolis"),
                    reaction_rate_ps=1.0 if mode == "rate" else None)
    result = lipf6_app.main(cfg)
    _, kb_md = md_unit_factors(unit_system)
    assert captured["kb_real"] == pytest.approx(kb_md)
    assert captured["md_neighbor_fn"] is neighbor
    assert captured["md_energy_fn"] is result["md_energy_fn"]
    factor = 23.0605489 if unit_system == "real" else 1.0
    assert float(captured["md_energy_fn"](
        result["init_positions"], neighbor
    )) == pytest.approx(2.0 * factor)
    if mode == "metropolis":
        assert captured["beta"] == pytest.approx(
            1.0 / (8.617333262e-5 * cfg.temperature_k)
        )
