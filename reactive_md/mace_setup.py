from flax import nnx
from mace.calculators import foundations_models
from mace.tools.scripts_utils import extract_config_mace_model
from mace_jax.tools.device import configure_torch_runtime, get_torch_device
from jax_md._nn.mace.mace_jax_from_torch import convert_model


def load_foundation_model(source="mp", variant=None, device="cpu"):
    loader_kwargs = {"device": device}
    src = source.lower()

    if src in {"mp", "off", "omol"}:
        loader = getattr(foundations_models, f"mace_{src}")
        if variant is not None:
            loader_kwargs["model"] = variant
    elif src == "anicc":
        loader = foundations_models.mace_anicc
        if variant is not None:
            loader_kwargs["model_path"] = variant
    else:
        raise ValueError(f"Unknown foundation source: {source!r}")

    return loader(return_raw_model=True, **loader_kwargs).float().eval()


def build_mace_model(source="mp", variant="small-0b2", *, return_backend_id=False):
    torch_device = configure_torch_runtime(get_torch_device(), deterministic=True)

    torch_model = load_foundation_model(source, variant, device="cpu")
    torch_config = extract_config_mace_model(torch_model)

    if return_backend_id:
        from .mace_identity import fingerprint
        from importlib.metadata import version
        versions = {name: version(name) for name in
                    ("mace-torch", "mace-jax", "jax", "jax-md", "flax", "torch")}
        model_id = fingerprint(source.lower(), variant, versions,
                               type(torch_model), repr(torch_model), torch_config,
                               torch_model.state_dict())

    torch_model = torch_model.to(torch_device)

    graphdef, nnx_state, jax_config = convert_model(torch_model, torch_config)
    jax_model = nnx.merge(graphdef, nnx_state)

    if return_backend_id:
        # Include the converted parameters and configuration, not only source weights.
        model_id = fingerprint(model_id, jax_config, nnx.state(jax_model).to_pure_dict())
        return jax_model, jax_config, torch_config, model_id
    return jax_model, jax_config, torch_config
