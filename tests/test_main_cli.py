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

