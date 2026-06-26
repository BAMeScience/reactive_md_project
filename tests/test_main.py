import types

import jax.numpy as jnp
import numpy as np

from reactive_md.config import SimConfig
import reactive_md.main as main_mod


def _empty_bonds():
    return (
        np.zeros((0, 2), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


def _empty_angles():
    return (
        np.zeros((0, 3), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.float32),
    )


def _empty_dihedrals():
    return (
        np.zeros((0, 4), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
        np.zeros((0,), dtype=np.int32),
        np.zeros((0,), dtype=np.float32),
    )


def _fake_parse_lammps_data(data_file, settings_file):
    """Minimal parser output with one PF6 unit and one Li atom.

    Atom layout:
        0      P
        1..6   F
        7      Li
    """

    positions = np.array(
        [
            [0.0, 0.0, 0.0],  # P
            [1.6, 0.0, 0.0],  # F
            [-1.6, 0.0, 0.0],
            [0.0, 1.6, 0.0],
            [0.0, -1.6, 0.0],
            [0.0, 0.0, 1.6],
            [0.0, 0.0, -1.6],
            [3.0, 0.0, 0.0],  # Li
        ],
        dtype=np.float32,
    )

    bonds = _empty_bonds()
    angles = _empty_angles()
    torsions = _empty_dihedrals()
    impropers = _empty_dihedrals()

    charges = np.zeros((8,), dtype=np.float32)
    sigmas = np.full((8,), 3.0, dtype=np.float32)
    epsilons = np.full((8,), 0.1, dtype=np.float32)
    nonbonded = (charges, sigmas, epsilons, None, None)

    molecule_id = np.array([1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32)
    box = jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32)
    masses = np.ones((8,), dtype=np.float32)

    # Match default SimConfig: p_type=6, f_type=7, li_type=8.
    atom_types = np.array([6, 7, 7, 7, 7, 7, 7, 8], dtype=np.int32)

    return (
        positions,
        bonds,
        angles,
        torsions,
        impropers,
        nonbonded,
        molecule_id,
        box,
        masses,
        atom_types,
    )


def test_main_smoke_with_mocked_parser(monkeypatch):
    """main() should still run with mocked I/O and no MD production work."""

    monkeypatch.setattr(main_mod, "parse_lammps_data", _fake_parse_lammps_data)

    captured = {}

    def fake_build_forcefield(**kwargs):
        captured["build_forcefield_kwargs"] = kwargs

        def disp_fn(a, b):
            return b - a

        neighbor_fn = types.SimpleNamespace(
            allocate=lambda R: object(),
            update=lambda R, nlist: nlist,
        )

        def energy_fn(R, nlist):
            return {"total": jnp.array(0.0)}

        return types.SimpleNamespace(
            disp_fn=disp_fn,
            neighbor_fn=neighbor_fn,
            energy_fn=energy_fn,
            nlist=object(),
            nb_options=types.SimpleNamespace(r_cut=15.0, dr_threshold=0.5),
        )

    def fake_run_md_nvt_with_reactions(**kwargs):
        captured["run_kwargs"] = kwargs
        return types.SimpleNamespace(accepted_events=0)

    monkeypatch.setattr(main_mod, "build_forcefield", fake_build_forcefield)
    monkeypatch.setattr(
        main_mod,
        "run_md_nvt_with_reactions",
        fake_run_md_nvt_with_reactions,
    )

    cfg = SimConfig(
        data_file="dummy.data",
        settings_file="dummy.settings",
        steps=2,
        check_every=1,
        max_events=0,
        sigma_mid=0.1,
        sigma_width=0.3,
    )

    main_mod.main(cfg)

    assert "build_forcefield_kwargs" in captured
    assert "run_kwargs" in captured
    assert captured["run_kwargs"]["cfg"] is cfg
    assert callable(captured["run_kwargs"]["reaction_step_fn"])


def test_reaction_step_fn_metropolis_passes_sigma_parameters(monkeypatch):
    """The closure created in main() should pass sigma_mid/sigma_width onward."""

    monkeypatch.setattr(main_mod, "parse_lammps_data", _fake_parse_lammps_data)

    def fake_build_forcefield(**kwargs):
        def disp_fn(a, b):
            return b - a

        neighbor_fn = types.SimpleNamespace(
            allocate=lambda R: object(),
            update=lambda R, nlist: nlist,
        )

        def energy_fn(R, nlist):
            return {"total": jnp.array(0.0)}

        return types.SimpleNamespace(
            disp_fn=disp_fn,
            neighbor_fn=neighbor_fn,
            energy_fn=energy_fn,
            nlist=object(),
            nb_options=types.SimpleNamespace(r_cut=15.0, dr_threshold=0.5),
        )

    captured = {}

    def fake_maybe_react_one_event(*args, **kwargs):
        captured["metropolis_kwargs"] = kwargs
        return args[0], False, kwargs["ff"], kwargs["sys"], {"mode": "metropolis"}, args[1]

    def fake_run_md_nvt_with_reactions(**kwargs):
        reaction_step_fn = kwargs["reaction_step_fn"]
        key = jnp.array([0, 0], dtype=jnp.uint32)
        R = jnp.zeros((8, 3), dtype=jnp.float32)
        ff = kwargs["ff"]
        sys = kwargs["sys"]
        reaction_step_fn(key, R, ff, sys)
        return types.SimpleNamespace(accepted_events=0)

    monkeypatch.setattr(main_mod, "build_forcefield", fake_build_forcefield)
    monkeypatch.setattr(main_mod, "maybe_react_one_event", fake_maybe_react_one_event)
    monkeypatch.setattr(
        main_mod,
        "run_md_nvt_with_reactions",
        fake_run_md_nvt_with_reactions,
    )

    cfg = SimConfig(
        data_file="dummy.data",
        settings_file="dummy.settings",
        reaction_mode="metropolis",
        sigma_mid=0.25,
        sigma_width=0.05,
    )

    main_mod.main(cfg)

    kwargs = captured["metropolis_kwargs"]
    assert kwargs["sigma_mid"] == 0.25
    assert kwargs["sigma_width"] == 0.05
    assert "r_lif_on" not in kwargs
    assert "r_pf_break" not in kwargs
    assert "thermo_gate_mode" not in kwargs
    assert "thermo_gate_coordinate" not in kwargs


def test_reaction_step_fn_rate_passes_sigma_parameters(monkeypatch):
    """Rate mode should also use sigma_mid/sigma_width, not old rate_pf_* gates."""

    monkeypatch.setattr(main_mod, "parse_lammps_data", _fake_parse_lammps_data)

    def fake_build_forcefield(**kwargs):
        def disp_fn(a, b):
            return b - a

        neighbor_fn = types.SimpleNamespace(
            allocate=lambda R: object(),
            update=lambda R, nlist: nlist,
        )

        def energy_fn(R, nlist):
            return {"total": jnp.array(0.0)}

        return types.SimpleNamespace(
            disp_fn=disp_fn,
            neighbor_fn=neighbor_fn,
            energy_fn=energy_fn,
            nlist=object(),
            nb_options=types.SimpleNamespace(r_cut=15.0, dr_threshold=0.5),
        )

    captured = {}

    def fake_maybe_react_rate_events(*args, **kwargs):
        captured["rate_kwargs"] = kwargs
        return args[0], False, kwargs["ff"], kwargs["sys"], {"mode": "rate"}, args[1]

    def fake_run_md_nvt_with_reactions(**kwargs):
        reaction_step_fn = kwargs["reaction_step_fn"]
        key = jnp.array([0, 0], dtype=jnp.uint32)
        R = jnp.zeros((8, 3), dtype=jnp.float32)
        ff = kwargs["ff"]
        sys = kwargs["sys"]
        reaction_step_fn(key, R, ff, sys)
        return types.SimpleNamespace(accepted_events=0)

    monkeypatch.setattr(main_mod, "build_forcefield", fake_build_forcefield)
    monkeypatch.setattr(main_mod, "maybe_react_rate_events", fake_maybe_react_rate_events)
    monkeypatch.setattr(
        main_mod,
        "run_md_nvt_with_reactions",
        fake_run_md_nvt_with_reactions,
    )

    cfg = SimConfig(
        data_file="dummy.data",
        settings_file="dummy.settings",
        reaction_mode="rate",
        sigma_mid=-0.1,
        sigma_width=0.15,
        reaction_rate_ps=1.0,
    )

    main_mod.main(cfg)

    kwargs = captured["rate_kwargs"]
    assert kwargs["sigma_mid"] == -0.1
    assert kwargs["sigma_width"] == 0.15
    assert "r_lif_on" not in kwargs
    assert "r_pf_break" not in kwargs
    assert "rate_pf_mode" not in kwargs
    assert "rate_gate_coordinate" not in kwargs

