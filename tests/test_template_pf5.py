import numpy as np
from reactive_md.templates_pf5 import make_pf5_template, make_lif_template

def test_pf5_template_shapes_and_units():
    pf5 = make_pf5_template()

    assert pf5.bond_idx_local.shape == (5, 2)
    assert pf5.angle_idx_local.shape == (10, 3)
    assert pf5.k_b.shape == (5,)
    assert pf5.r0.shape == (5,)
    assert pf5.k_theta.shape == (10,)
    assert pf5.theta0.shape == (10,)

    # theta0 should be in radians
    assert np.all(pf5.theta0 > 0.0)
    assert np.all(pf5.theta0 < np.pi + 1e-6)

    # PF5 pair and charges exist
    assert "P" in pf5.pair and "F" in pf5.pair
    assert "P" in pf5.q and "F" in pf5.q

def test_lif_template_contents():
    lif = make_lif_template()
    assert "Li" in lif.nb and "F" in lif.nb
    for k in ("q", "eps", "sigma"):
        assert k in lif.nb["Li"]
        assert k in lif.nb["F"]

