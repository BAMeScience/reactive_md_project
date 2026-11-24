"""MACE adapter contracts without loading foundation weights."""
from types import SimpleNamespace
from unittest.mock import Mock, call

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture
def energy_module():
    pytest.importorskip("mace")
    from reactive_md import mace_energy
    return mace_energy


@pytest.fixture
def setup_module():
    pytest.importorskip("mace")
    pytest.importorskip("mace_jax")
    from reactive_md import mace_setup
    return mace_setup


@pytest.mark.parametrize("output", [
    3.0,
    [1.0, 2.0],
    {"energy": [1.0, 2.0]},
    {"energies": [1.0, 2.0]},
    {"energy": [1.0, 2.0], "energies": [100.0]},
])
def test_model_energy_scalar_reduction(energy_module, output):
    if isinstance(output, dict):
        output = {k: jnp.asarray(v) for k, v in output.items()}
    else:
        output = jnp.asarray(output)
    result = energy_module.model_energy(output)
    assert result.shape == ()
    assert float(result) == pytest.approx(3.0)


def test_model_energy_requires_energy_key(energy_module):
    with pytest.raises(KeyError, match="No energy key found"):
        energy_module.model_energy({"forces": jnp.zeros((2, 3))})


@pytest.mark.parametrize("ensemble,fractional", [("nve", False), ("npt", True)])
def test_energy_system_forwards_geometry_and_is_differentiable(
    energy_module, monkeypatch, ensemble, fractional
):
    mod = energy_module
    box0 = jnp.eye(3) * 10
    box1 = jnp.eye(3) * 12
    perturbation = jnp.eye(3)
    positions = jnp.array([[1., 2., 3.], [-1., 0., 2.]])
    z = np.array([3, 9], dtype=np.int32)
    config = {"r_max": 4.5}
    disp, shift, neighbor_fn, neighbors = object(), object(), object(), object()
    periodic = Mock(return_value=(disp, shift))
    partition = Mock(return_value=neighbor_fn)
    calls = []

    def featurize(R, nbrs, *, box, perturbation):
        assert nbrs is neighbors
        calls.append((box, perturbation))
        return R

    factory = Mock(return_value=featurize)
    monkeypatch.setattr(mod.space, "periodic_general", periodic)
    monkeypatch.setattr(mod.partition, "neighbor_list", partition)
    monkeypatch.setattr(mod, "mace_featurizer", factory)

    def model(batch, *, compute_stress):
        assert compute_stress is False
        return {"energies": jnp.sum(batch ** 2, axis=-1)}

    d, s, n, make_energy = mod.build_mace_energy_system(
        jax_model=model, jax_model_config=config, z_atomic=z, box0=box0,
        ensemble=ensemble, dr_threshold=0.3, capacity_multiplier=2.5,
    )
    assert (d, s, n) == (disp, shift, neighbor_fn)
    args, kwargs = periodic.call_args
    assert args[0] is box0
    assert kwargs == {"fractional_coordinates": fractional}
    args, kwargs = partition.call_args
    assert args[0] is disp and args[1] is box0 and args[2] == 4.5
    assert kwargs == dict(dr_threshold=0.3, capacity_multiplier=2.5,
                          format=mod.partition.Dense,
                          fractional_coordinates=fractional)
    args, kwargs = factory.call_args
    assert args[0] is disp and args[1] is config and args[2] is z
    assert kwargs == {"fractional_coordinates": fractional}
    energy = make_energy(neighbors)
    assert float(energy(positions)) == pytest.approx(19.)
    assert calls[-1][0] is box0 and calls[-1][1] is None
    assert float(energy(positions, box=box1, perturbation=perturbation,
                        unused_keyword=True)) == pytest.approx(19.)
    assert calls[-1][0] is box1 and calls[-1][1] is perturbation
    np.testing.assert_allclose(jax.grad(energy)(positions), 2 * positions)


@pytest.mark.parametrize("source,variant", [
    ("mp", None), ("MP", "custom"), ("off", None), ("off", "custom"),
    ("omol", None), ("omol", "custom"), ("anicc", None), ("anicc", "custom"),
])
def test_foundation_loader_routes_variant(setup_module, monkeypatch, source, variant):
    mod = setup_module
    raw = Mock()
    raw.float.return_value = raw
    raw.eval.return_value = raw
    loader = Mock(return_value=raw)
    monkeypatch.setattr(mod.foundations_models, f"mace_{source.lower()}", loader,
                        raising=False)
    assert mod.load_foundation_model(source, variant, device="cpu") is raw
    expected = dict(device="cpu", return_raw_model=True)
    if variant is not None:
        expected["model_path" if source.lower() == "anicc" else "model"] = variant
    loader.assert_called_once_with(**expected)
    assert raw.method_calls == [call.float(), call.eval()]


def test_foundation_loader_rejects_unknown_source(setup_module):
    with pytest.raises(ValueError, match="Unknown foundation source"):
        setup_module.load_foundation_model("not-a-foundation")


def test_model_conversion_contract(setup_module, monkeypatch):
    mod = setup_module
    raw, moved = Mock(), object()
    raw.to.return_value = moved
    torch_config, jax_config = object(), object()
    graph, state, model = object(), object(), object()
    get_device = Mock(return_value="requested-device")
    configure = Mock(return_value="configured-device")
    load = Mock(return_value=raw)
    extract = Mock(return_value=torch_config)
    convert = Mock(return_value=(graph, state, jax_config))
    merge = Mock(return_value=model)
    for name, mock in [("get_torch_device", get_device),
                       ("configure_torch_runtime", configure),
                       ("load_foundation_model", load),
                       ("extract_config_mace_model", extract),
                       ("convert_model", convert)]:
        monkeypatch.setattr(mod, name, mock)
    monkeypatch.setattr(mod, "nnx", SimpleNamespace(merge=merge))
    assert mod.build_mace_model("off", "custom") == (model, jax_config, torch_config)
    get_device.assert_called_once_with()
    configure.assert_called_once_with("requested-device", deterministic=True)
    load.assert_called_once_with("off", "custom", device="cpu")
    extract.assert_called_once_with(raw)
    raw.to.assert_called_once_with("configured-device")
    convert.assert_called_once_with(moved, torch_config)
    merge.assert_called_once_with(graph, state)


def test_checkpoint_model_identity_includes_converted_weights(setup_module, monkeypatch):
    mod = setup_module
    raw = Mock()
    raw.state_dict.return_value = {'w': np.array([1.], dtype=np.float32)}
    raw.to.return_value = raw
    # Mock repr includes an address, but this string is constant within this test.
    weights = {'w': np.array([2.], dtype=np.float32)}
    state = SimpleNamespace(to_pure_dict=lambda: weights)
    monkeypatch.setattr(mod, 'get_torch_device', lambda: 'cpu')
    monkeypatch.setattr(mod, 'configure_torch_runtime', lambda *a, **kw: 'cpu')
    monkeypatch.setattr(mod, 'load_foundation_model', lambda *a, **kw: raw)
    monkeypatch.setattr(mod, 'extract_config_mace_model', lambda m: {'r_max': 5.})
    monkeypatch.setattr(mod, 'convert_model', lambda *a: (None, None, {'r_max': 5.}))
    monkeypatch.setattr(mod, 'nnx', SimpleNamespace(merge=lambda *a: None,
                                                   state=lambda m: state))
    monkeypatch.setattr('importlib.metadata.version', lambda name: 'test-version')
    first = mod.build_mace_model(return_backend_id=True)[3]
    assert mod.build_mace_model(return_backend_id=True)[3] == first
    weights['w'][0] = 3.
    assert mod.build_mace_model(return_backend_id=True)[3] != first
