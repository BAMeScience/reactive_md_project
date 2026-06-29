"""Configuration tests for the sigma-only reaction interface."""

from __future__ import annotations

from dataclasses import fields

from reactive_md.config import SimConfig


def test_config_has_sigma_parameters():
    cfg = SimConfig()

    assert hasattr(cfg, "sigma_mid")
    assert hasattr(cfg, "sigma_width")
    assert cfg.sigma_width > 0.0


def test_config_no_longer_exposes_reaction_distance_gates():
    names = {field.name for field in fields(SimConfig)}

    assert "r_lif_on" not in names
    assert "r_pf_break" not in names
    assert "rate_pf_mode" not in names
    assert "rate_pf_mid" not in names
    assert "rate_pf_width" not in names


def test_r_pf_probe_is_still_available_for_product_geometry():
    cfg = SimConfig()

    assert hasattr(cfg, "r_pf_probe")
    assert cfg.r_pf_probe > 0.0


def test_config_override_sigma_parameters():
    cfg = SimConfig(sigma_mid=0.1, sigma_width=0.05)

    assert cfg.sigma_mid == 0.1
    assert cfg.sigma_width == 0.05

