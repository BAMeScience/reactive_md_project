from reactive_md.config import SimConfig

def test_config_defaults():
    cfg = SimConfig()
    assert cfg.steps == 2000
    assert cfg.check_every == 50
    assert cfg.r_cut == 15.0
    assert cfg.p_type == 6
    assert cfg.f_type == 7
    assert cfg.li_type == 8

def test_config_override():
    cfg = SimConfig(steps=10, temperature_k=350.0, r_on=3.2)
    assert cfg.steps == 10
    assert cfg.temperature_k == 350.0
    assert cfg.r_on == 3.2

