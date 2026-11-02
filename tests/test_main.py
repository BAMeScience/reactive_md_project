import numpy as np
import jax.numpy as jnp
import types

from reactive_md.config import SimConfig
import reactive_md.main as main_mod

def test_main_smoke_with_mocked_parser(monkeypatch):
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

    monkeypatch.setattr(main_mod, "parse_lammps_data", fake_parse_lammps_data)

    # prevent MD from running long: make steps very small
    cfg = SimConfig(steps=2, check_every=1, max_events=0)
    # Should not raise
    main_mod.main(cfg)

