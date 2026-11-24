"""CLI smoke tests for the sigma-only command-line interface."""

from __future__ import annotations

import pytest

from reactive_md.main import cli


def test_cli_help_mentions_sigma_options(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["reactive-md", "--help"])

    with pytest.raises(SystemExit) as excinfo:
        cli()

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    assert "--sigma-mid" in help_text
    assert "--sigma-width" in help_text


def test_cli_help_does_not_expose_legacy_reaction_gates(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["reactive-md", "--help"])

    with pytest.raises(SystemExit):
        cli()

    help_text = capsys.readouterr().out
    assert "--r-lif-on" not in help_text
    assert "--r-pf-break" not in help_text
    assert "--rate-pf-mode" not in help_text
    assert "--rate-pf-mid" not in help_text
    assert "--rate-pf-width" not in help_text



@pytest.mark.parametrize('mode,flags,mc,md', [
    ('metropolis', [], False, False),
    ('metropolis', ['--use-mace-mc'], True, False),
    ('metropolis', ['--use-mace-mc', '--use-mace-md'], True, True),
    ('rate', [], False, False),
    ('rate', ['--use-mace-md'], False, True),
])
def test_cli_forwards_supported_modes(monkeypatch, tmp_path, mode, flags, mc, md):
    from reactive_md.apps import lipf6

    calls = []
    monkeypatch.setattr(lipf6, 'main', lambda cfg, **kw: calls.append((cfg, kw)))
    cli(['lipf6', '--data', str(tmp_path / 'input.data'),
         '--settings', str(tmp_path / 'input.settings'), '--reaction-mode', mode,
         '--steps', '12', '--check-every', '3', '--seed', '17',
         '--mace-source', 'mp', '--mace-variant', 'small-0b2',
         '--reaction-rate-ps', '2', *flags])
    assert len(calls) == 1
    cfg, options = calls[0]
    assert (cfg.reaction_mode, cfg.use_mace_mc, cfg.use_mace_md) == (mode, mc, md)
    assert (cfg.steps, cfg.check_every, cfg.prng_seed) == (12, 3, 17)
    assert cfg.mace_variant == 'small-0b2'
    assert cfg.reaction_rate_ps == 2.
    assert cfg.data_file == str(tmp_path / 'input.data')
    assert cfg.settings_file == str(tmp_path / 'input.settings')
    assert options == dict(checkpoint_path=None, checkpoint_every_steps=None, restart_path=None)


@pytest.mark.parametrize('prefix,target', [([], 'lipf6'), (['lipf6'], 'lipf6'), (['a2'], 'a2')])
def test_unified_cli_routes_arguments(monkeypatch, prefix, target):
    from reactive_md.apps import lipf6, a2

    calls = []
    monkeypatch.setattr(lipf6, 'cli', lambda args: calls.append(('lipf6', args)))
    monkeypatch.setattr(a2, 'cli', lambda args: calls.append(('a2', args)))
    cli(prefix + ['--help'])
    assert calls == [(target, ['--help'])]
