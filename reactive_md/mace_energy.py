import jax.numpy as jnp
from jax_md import space, partition
from jax_md._nn.mace.featurizer import mace_featurizer


def model_energy(output):
    if isinstance(output, dict):
        if "energy" in output:
            return jnp.sum(output["energy"])
        if "energies" in output:
            return jnp.sum(output["energies"])
        raise KeyError(f"No energy key found: {output.keys()}")
    return jnp.sum(output)


def build_mace_energy_system(
    *,
    jax_model,
    jax_model_config,
    z_atomic,
    box0,
    ensemble,
    dr_threshold=0.5,
    capacity_multiplier=6.0,
):
    use_fractional = ensemble == "npt"

    displacement_fn, shift_fn = space.periodic_general(
        box0,
        fractional_coordinates=use_fractional,
    )

    neighbor_fn = partition.neighbor_list(
        displacement_fn,
        box0,
        float(jax_model_config["r_max"]),
        dr_threshold=dr_threshold,
        capacity_multiplier=capacity_multiplier,
        format=partition.Dense,
        fractional_coordinates=use_fractional,
    )

    featurize = mace_featurizer(
        displacement_fn,
        jax_model_config,
        z_atomic,
        fractional_coordinates=use_fractional,
    )

    def make_energy_fn(neighbors):
        def energy_fn(R, box=None, perturbation=None, **kwargs):
            del kwargs
            if box is None:
                box = box0

            batch = featurize(
                R,
                neighbors,
                box=box,
                perturbation=perturbation,
            )
            return model_energy(jax_model(batch, compute_stress=False))

        return energy_fn

    return displacement_fn, shift_fn, neighbor_fn, make_energy_fn
