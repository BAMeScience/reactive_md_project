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
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.6, 0.0, 0.0],
            [-1.6, 0.0, 0.0],
            [0.0, 1.6, 0.0],
            [0.0, -1.6, 0.0],
            [0.0, 0.0, 1.6],
            [0.0, 0.0, -1.6],
            [3.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    charges = np.zeros((8,), dtype=np.float32)
    sigmas = np.full((8,), 3.0, dtype=np.float32)
    epsilons = np.full((8,), 0.1, dtype=np.float32)
    nonbonded = (charges, sigmas, epsilons, None, None)

    return (
        positions,
        _empty_bonds(),
        _empty_angles(),
        _empty_dihedrals(),
        _empty_dihedrals(),
        nonbonded,
        np.array([1, 1, 1, 1, 1, 1, 1, 2], dtype=np.int32),
        jnp.array([20.0, 20.0, 20.0], dtype=jnp.float32),
        np.ones((8,), dtype=np.float32),
        np.array([6, 7, 7, 7, 7, 7, 7, 8], dtype=np.int32),
    )


def _fake_build_forcefield(**kwargs):
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


def test_main_smoke_with_mocked_parser(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_lammps_data", _fake_parse_lammps_data)

    captured = {}

    def fake_run_md_nvt_with_reactions(key, **kwargs):
        captured["driver_key"] = key
        captured["run_kwargs"] = kwargs
        return types.SimpleNamespace(accepted_events=0)

    monkeypatch.setattr(main_mod, "build_forcefield", _fake_build_forcefield)
    monkeypatch.setattr(main_mod, "run_md_nvt_with_reactions", fake_run_md_nvt_with_reactions)

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

    assert "run_kwargs" in captured
    assert captured["run_kwargs"]["cfg"] is cfg
    assert callable(captured["run_kwargs"]["reaction_step_fn"])


def test_reaction_step_fn_metropolis_passes_sigma_probability_parameters(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_lammps_data", _fake_parse_lammps_data)

    captured = {}

    def fake_maybe_react_one_event(*args, **kwargs):
        captured["metropolis_kwargs"] = kwargs
        return args[0], False, kwargs["ff"], kwargs["sys"], {"mode": "metropolis"}, args[1]

    def fake_run_md_nvt_with_reactions(key, **kwargs):
        reaction_step_fn = kwargs["reaction_step_fn"]
        R = jnp.zeros((8, 3), dtype=jnp.float32)
        reaction_step_fn(key, R, kwargs["ff"], kwargs["sys"])
        return types.SimpleNamespace(accepted_events=0)

    monkeypatch.setattr(main_mod, "build_forcefield", _fake_build_forcefield)
    monkeypatch.setattr(main_mod, "maybe_react_one_event", fake_maybe_react_one_event)
    monkeypatch.setattr(main_mod, "run_md_nvt_with_reactions", fake_run_md_nvt_with_reactions)

    cfg = SimConfig(
        data_file="dummy.data",
        settings_file="dummy.settings",
        reaction_mode="metropolis",
        sigma_mid=0.25,
        sigma_width=0.05,
    )

    main_mod.main(cfg)

    kwargs = captured["metropolis_kwargs"]
    assert kwargs["r_pf_probe"] == cfg.r_pf_probe
    assert "beta" in kwargs
    assert "mc_energy_evaluator" in kwargs
    assert "candidate_log_top_n" in kwargs

    assert kwargs["sigma_mid"] == 0.25
    assert kwargs["sigma_width"] == 0.05
    assert "r_lif_on" not in kwargs
    assert "r_pf_break" not in kwargs
    assert "thermo_gate_mode" not in kwargs
    assert "thermo_gate_coordinate" not in kwargs


def test_reaction_step_fn_rate_still_passes_sigma_probability_parameters(monkeypatch):
    monkeypatch.setattr(main_mod, "parse_lammps_data", _fake_parse_lammps_data)

    captured = {}

    def fake_maybe_react_rate_events(*args, **kwargs):
        captured["rate_kwargs"] = kwargs
        return args[0], False, kwargs["ff"], kwargs["sys"], {"mode": "rate"}, args[1]

    def fake_run_md_nvt_with_reactions(key, **kwargs):
        reaction_step_fn = kwargs["reaction_step_fn"]
        R = jnp.zeros((8, 3), dtype=jnp.float32)
        reaction_step_fn(key, R, kwargs["ff"], kwargs["sys"])
        return types.SimpleNamespace(accepted_events=0)

    monkeypatch.setattr(main_mod, "build_forcefield", _fake_build_forcefield)
    monkeypatch.setattr(main_mod, "maybe_react_rate_events", fake_maybe_react_rate_events)
    monkeypatch.setattr(main_mod, "run_md_nvt_with_reactions", fake_run_md_nvt_with_reactions)

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
    assert kwargs["reaction_rate_ps"] == 1.0
    assert kwargs["reactive_interval_ps"] == cfg.check_every * cfg.dt

    assert "r_lif_on" not in kwargs
    assert "r_pf_break" not in kwargs
    assert "rate_pf_mode" not in kwargs
    assert "rate_gate_coordinate" not in kwargs

