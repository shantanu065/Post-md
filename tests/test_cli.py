"""CLI smoke tests (help and command discovery only — no real trajectory needed)."""

from __future__ import annotations

from typer.testing import CliRunner

from post_md.cli.main import app


def test_cli_help_lists_all_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for cmd in ("info", "rmsd", "rmsf", "rg", "pca", "cluster"):
        assert cmd in result.stdout
