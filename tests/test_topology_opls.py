import numpy as np
import pytest
from reactive_md.topology_opls import (
    discover_pf6_and_li,
    embed_pf5_into_pf6,
    remove_terms_in_molid,
)
from reactive_md.templates_pf5 import make_pf5_template

def test_discover_pf6_and_li_simple():
    # 8 atoms: PF6 (0..6), Li (7)
    atom_types = np.array([6,7,7,7,7,7,7,8], dtype=int)
    molecule_id = np.array([1,1,1,1,1,1,1,2], dtype=int)

    pf6_atoms, li_atoms = discover_pf6_and_li(atom_types, molecule_id, p_type=6, f_type=7, li_type=8)

    assert pf6_atoms.shape == (1, 7)
    assert pf6_atoms[0,0] == 0
    assert set(pf6_atoms[0,1:].tolist()) == set([1,2,3,4,5,6])
    assert li_atoms.tolist() == [7]

def test_embed_pf5_into_pf6():
    pf5 = make_pf5_template()
    pf6_block = np.array([0,1,2,3,4,5,6], dtype=np.int32)
    leave_F = 6

    pf5_glob, bonds_g, angles_g = embed_pf5_into_pf6(
        pf6_block, leave_F,
        pf5_bond_idx_local=pf5.bond_idx_local,
        pf5_angle_idx_local=pf5.angle_idx_local
    )

    assert pf5_glob.shape == (6,)  # P + 5F
    assert pf5_glob[0] == 0
    assert leave_F not in pf5_glob[1:]
    assert bonds_g.shape == (5, 2)
    assert angles_g.shape == (10, 3)

def test_embed_pf5_invalid_leaving_f_raises():
    pf5 = make_pf5_template()
    pf6_block = np.array([0,1,2,3,4,5,6], dtype=np.int32)
    with pytest.raises(ValueError):
        embed_pf5_into_pf6(
            pf6_block, leave_F=99,
            pf5_bond_idx_local=pf5.bond_idx_local,
            pf5_angle_idx_local=pf5.angle_idx_local
        )

def test_remove_terms_in_molid():
    # Two bonds: (0,1) inside molid=1; (0,7) crosses molids
    term_idx = np.array([[0,1],[0,7]], dtype=np.int32)
    molecule_id = np.array([1,1,1,1,1,1,1,2], dtype=np.int32)

    k = np.array([100.0, 200.0], dtype=np.float32)
    r0 = np.array([1.0, 2.0], dtype=np.float32)

    term2, (k2, r02) = remove_terms_in_molid(term_idx, [k,r0], molecule_id, molid=1)
    assert term2.shape == (1,2)
    assert term2[0].tolist() == [0,7]
    assert k2.tolist() == [200.0]
    assert r02.tolist() == [2.0]

