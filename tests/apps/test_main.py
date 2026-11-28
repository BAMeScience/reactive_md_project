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
        if captured:
            assert kwargs["relaxation_kernel_cache"] is captured["relaxation_kernel_cache"]
        captured.update(kwargs)

    def driver_stub(key, **kwargs):
        R = kwargs["init_positions"]
        kwargs["reaction_step_fn"](
            key, R, kwargs["ff"], kwargs["sys"], jnp.zeros_like(R),
            interval_ps=kwargs["cfg"].check_every * kwargs["cfg"].dt,
        )
        kwargs["reaction_step_fn"](key, R, kwargs["ff"], kwargs["sys"],
                                   jnp.zeros_like(R), interval_ps=.01)
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
    from reactive_md.relaxation import LangevinKernelCache
    assert isinstance(captured["relaxation_kernel_cache"], LangevinKernelCache)
    factor = 23.0605489 if unit_system == "real" else 1.0
    assert float(captured["md_energy_fn"](
        result["init_positions"], neighbor
    )) == pytest.approx(2.0 * factor)
    if mode == "metropolis":
        assert captured["beta"] == pytest.approx(
            1.0 / (8.617333262e-5 * cfg.temperature_k)
        )


def test_rate_application_uses_actual_chunk_interval(monkeypatch, mocked_parser):
    from reactive_md.engine.contracts import ReactionContext, MDRunResult
    from reactive_md.reactions import lipf6_driver

    intervals = []

    def fake_generic_run(key, **kwargs):
        cfg = kwargs['cfg']
        R = kwargs['init_positions']
        ff, system = kwargs['ff'], kwargs['sys']
        state = types.SimpleNamespace(position=R, velocity=jnp.zeros_like(R))
        for step, interval in [(2, .02), (4, .02), (5, .01)]:
            outcome = kwargs['reaction_handler'].attempt(ReactionContext(
                key=key, step=step, interval_ps=interval, positions=R,
                velocities=state.velocity, system=system, ff=ff,
                box=kwargs['box'], shift_fn=kwargs['shift_fn'],
                masses=kwargs['masses'], temperature_k=cfg.temperature_k,
                kb_real=cfg.kb_real,
            ))
            intervals.append(outcome.metadata['dt_reactive_ps'])
            assert outcome.metadata['k_rate_ps'] == 20.0
            assert not outcome.events
            key = outcome.key
        return MDRunResult(state, ff, system, 0, 5, key, 0.)

    monkeypatch.setattr(lipf6_driver, 'run_md_nvt', fake_generic_run)
    lipf6_app.main(SimConfig(reaction_mode='rate', reaction_rate_ps=20.,
                            steps=5, check_every=2, dt=.01))
    assert intervals == pytest.approx([.02, .02, .01])


@pytest.mark.parametrize('mode,mc,md', [
    ('metropolis', False, False),
    ('metropolis', True, False),
    ('metropolis', True, True),
    ('rate', False, False),
    ('rate', False, True),
])
def test_supported_application_modes(monkeypatch, mocked_parser, mode, mc, md):
    from reactive_md.mace_mc_energy import MaceJaxEnergyEvaluator

    neighbor = object()
    model = object()
    calls = []
    setup = types.ModuleType('reactive_md.mace_setup')
    energy = types.ModuleType('reactive_md.mace_energy')

    def build_model(**kwargs):
        calls.append(('model', kwargs))
        return model, {}, None

    def build_energy(**kwargs):
        calls.append(('energy', kwargs))
        return None, None, neighbor, lambda nl: lambda R, **kw: jnp.asarray(2.)

    setup.build_mace_model = build_model
    energy.build_mace_energy_system = build_energy
    monkeypatch.setitem(sys.modules, setup.__name__, setup)
    monkeypatch.setitem(sys.modules, energy.__name__, energy)
    captured = []

    def reaction_stub(name):
        def run(*args, **kwargs):
            captured.append((name, kwargs))
        return run

    monkeypatch.setattr(lipf6_app, 'maybe_react_one_event', reaction_stub('metropolis'))
    monkeypatch.setattr(lipf6_app, 'maybe_react_rate_events', reaction_stub('rate'))

    def driver(key, **kwargs):
        R = kwargs['init_positions']
        kwargs['reaction_step_fn'](key, R, kwargs['ff'], kwargs['sys'],
                                   jnp.zeros_like(R), interval_ps=.01)
        return kwargs

    monkeypatch.setattr(lipf6_app, 'run_md_nvt_with_reactions', driver)
    cfg = SimConfig(reaction_mode=mode, use_mace_mc=mc, use_mace_md=md,
                    reaction_rate_ps=2. if mode == 'rate' else None,
                    mace_source='mp', mace_variant='small-0b2',
                    mace_dr_threshold=.7, mace_capacity_multiplier=2.)
    result = lipf6_app.main(cfg)
    assert len(captured) == 1
    name, kw = captured[0]
    assert name == mode
    assert (result['md_energy_fn'] is not None) == md
    assert result['md_neighbor_fn'] is (neighbor if md else None)
    assert kw['md_energy_fn'] is result['md_energy_fn']
    assert kw['md_neighbor_fn'] is result['md_neighbor_fn']
    assert (kw['relaxation_kernel_cache'] is not None) == md
    if mode == 'metropolis':
        evaluator = kw['mc_energy_evaluator']
        if mc:
            assert isinstance(evaluator, MaceJaxEnergyEvaluator)
            assert evaluator.neighbor_fn is neighbor
        else:
            assert evaluator is None
        kb = 8.617333262e-5 if mc else cfg.kb_real
        assert kw['beta'] == pytest.approx(1. / (kb * cfg.temperature_k))
    else:
        assert 'mc_energy_evaluator' not in kw
        assert kw['reactive_interval_ps'] == .01
    if mc or md:
        assert [name for name, _ in calls] == ['model', 'energy']
        assert calls[0][1] == {'source': cfg.mace_source, 'variant': cfg.mace_variant}
        options = calls[1][1]
        assert options['jax_model'] is model
        assert options['ensemble'] == 'nve'
        assert options['dr_threshold'] == .7
        assert options['capacity_multiplier'] == 2.
        np.testing.assert_array_equal(options['z_atomic'], [6, 6])
    else:
        assert calls == []


@pytest.mark.parametrize('mode,mc,md,message', [
    ('metropolis', False, True, 'requires MACE-MC'),
    ('rate', True, False, 'only supported in metropolis'),
    ('rate', True, True, 'only supported in metropolis'),
    ('unknown', False, False, 'Unsupported reaction mode'),
])
def test_unsupported_application_modes_fail_before_io(monkeypatch, mode, mc, md, message):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid mode reached input parsing')

    monkeypatch.setattr(lipf6_app, 'parse_lammps_data', forbidden)
    with pytest.raises(ValueError, match=message):
        lipf6_app.main(SimConfig(reaction_mode=mode, use_mace_mc=mc, use_mace_md=md))


@pytest.mark.parametrize('mode', ['metropolis', 'rate'])
def test_mace_checkpoint_identity_forwarded_to_generic_driver(
    monkeypatch, mocked_parser, tmp_path, mode
):
    from dataclasses import replace
    from reactive_md.engine.contracts import MDRunResult
    from reactive_md.reactions import lipf6_driver

    setup = types.ModuleType('reactive_md.mace_setup')
    energy = types.ModuleType('reactive_md.mace_energy')
    model_id = ['weights-a']
    calls = []

    def build(**kwargs):
        assert kwargs['return_backend_id'] is True
        return None, None, None, model_id[0]

    setup.build_mace_model = build
    energy.build_mace_energy_system = lambda **kw: (
        None, None, object(), lambda nl: lambda R, **kw: jnp.asarray(2.0)
    )
    monkeypatch.setitem(sys.modules, setup.__name__, setup)
    monkeypatch.setitem(sys.modules, energy.__name__, energy)

    def driver(key, **kwargs):
        calls.append(kwargs)
        return MDRunResult(None, kwargs['ff'], kwargs['sys'], 0, 0, key, 0.)

    monkeypatch.setattr(lipf6_driver, 'run_md_nvt', driver)
    cfg = SimConfig(reaction_mode=mode, use_mace_md=True,
                    use_mace_mc=(mode == 'metropolis'),
                    reaction_rate_ps=1. if mode == 'rate' else None)
    path = str(tmp_path / 'checkpoint')
    lipf6_app.main(cfg, checkpoint_path=path)
    original = calls[-1]['md_backend_id']
    assert original.startswith('mace-v1:')
    assert calls[-1]['checkpoint_path'] == path
    (tmp_path / 'checkpoint').touch()
    lipf6_app.main(replace(cfg, steps=cfg.steps + 100), restart_path=path)
    assert calls[-1]['md_backend_id'] == original
    assert calls[-1]['restart_path'] == path
    lipf6_app.main(replace(cfg, mace_dr_threshold=.9), checkpoint_path=path)
    assert calls[-1]['md_backend_id'] != original
    lipf6_app.main(replace(cfg, mace_energy_only=True), checkpoint_path=path)
    assert calls[-1]['md_backend_id'] != original
    model_id[0] = 'weights-b'
    lipf6_app.main(cfg, checkpoint_path=path)
    assert calls[-1]['md_backend_id'] != original


def test_energy_only_requires_mace_before_io(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Invalid mode reached input parsing')
    monkeypatch.setattr(lipf6_app, 'parse_lammps_data', forbidden)
    with pytest.raises(ValueError, match='requires a MACE backend'):
        lipf6_app.main(SimConfig(mace_energy_only=True))


def test_cli_energy_only_is_explicit(monkeypatch):
    configs = []
    monkeypatch.setattr(lipf6_app, 'main', lambda cfg, **kwargs: configs.append(cfg))
    lipf6_app.cli(['--data', 'data', '--settings', 'settings', '--use-mace-mc'])
    lipf6_app.cli(['--data', 'data', '--settings', 'settings', '--use-mace-mc', '--mace-energy-only'])
    assert [c.mace_energy_only for c in configs] == [False, True]
