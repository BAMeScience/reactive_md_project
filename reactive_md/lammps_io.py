import numpy as np
import pandas as pd


def read_lammps_system(data_file, settings_file, type_to_Z, parse_lammps_data):
    (
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
    ) = parse_lammps_data(data_file, settings_file)

    positions = np.asarray(positions)
    box = np.asarray(box)
    masses = np.asarray(masses)
    atom_types = np.asarray(atom_types, dtype=np.int32)

    z_atomic = np.array([type_to_Z[int(t)] for t in atom_types], dtype=np.int32)

    return {
        "positions": positions,
        "box": box,
        "masses": masses,
        "atom_types": atom_types,
        "z_atomic": z_atomic,
        "n_atoms": int(positions.shape[0]),
        "molecule_id": molecule_id,
        "bonds": bonds,
        "angles": angles,
        "torsions": torsions,
        "impropers": impropers,
        "nonbonded": nonbonded,
    }


def read_lammps_velocities(filename, n_atoms):
    df = pd.read_csv(
        filename,
        delimiter=r"\s+",
        header=None,
        skiprows=9,
        nrows=n_atoms,
        names=["id", "type", "vx", "vy", "vz"],
    )

    df = df.sort_values("id")
    return df[["vx", "vy", "vz"]].to_numpy(dtype=float)


def write_lammps_dump_frame(
    *,
    file,
    step,
    box,
    atom_ids,
    atom_types,
    masses,
    wrapped_positions,
    unwrapped_positions,
):
    n_atoms = len(atom_ids)

    file.write(f"ITEM: TIMESTEP\n{step}\n")
    file.write(f"ITEM: NUMBER OF ATOMS\n{n_atoms}\n")
    file.write("ITEM: BOX BOUNDS pp pp pp\n")

    box_arr = np.asarray(box)

    if box_arr.ndim == 1:
        file.write(f"0 {box_arr[0]}\n")
        file.write(f"0 {box_arr[1]}\n")
        file.write(f"0 {box_arr[2]}\n")
    elif box_arr.ndim == 2:
        file.write(f"0 {box_arr[0, 0]}\n")
        file.write(f"0 {box_arr[1, 1]}\n")
        file.write(f"0 {box_arr[2, 2]}\n")
    else:
        raise ValueError(f"Unexpected box shape: {box_arr.shape}")

    file.write("ITEM: ATOMS id type mass x y z xu yu zu\n")

    for i in range(n_atoms):
        file.write(
            f"{atom_ids[i]} {atom_types[i]} {masses[i]} "
            f"{wrapped_positions[i, 0]:.6f} "
            f"{wrapped_positions[i, 1]:.6f} "
            f"{wrapped_positions[i, 2]:.6f} "
            f"{unwrapped_positions[i, 0]:.6f} "
            f"{unwrapped_positions[i, 1]:.6f} "
            f"{unwrapped_positions[i, 2]:.6f}\n"
        )
