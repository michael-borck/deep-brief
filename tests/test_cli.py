"""Tests for CLI functionality."""

from typer.testing import CliRunner

from deep_brief.cli import app


def test_cli_help() -> None:
    """Test that CLI help works."""
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "DeepBrief" in result.stdout
    assert "Video Analysis Application" in result.stdout


def test_cli_analyze_help() -> None:
    """Test analyze command help."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "--help"])
    assert result.exit_code == 0
    assert "analyze" in result.stdout


def test_cli_no_args() -> None:
    """Test CLI with no arguments (web mode)."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze"])
    assert result.exit_code == 0
    assert "Launching web interface" in result.stdout


def test_cli_with_video() -> None:
    """Test CLI with video argument."""
    runner = CliRunner()
    result = runner.invoke(app, ["analyze", "test.mp4"])
    assert result.exit_code == 0
    assert "Analyzing video: test.mp4" in result.stdout


def test_cli_with_options() -> None:
    """Test CLI with various options."""
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "analyze",
            "test.mp4",
            "--output",
            "/tmp/output",
            "--config",
            "config.yaml",
            "--verbose",
        ],
    )
    assert result.exit_code == 0
    assert "Analyzing video: test.mp4" in result.stdout
    assert "Output directory: /tmp/output" in result.stdout
    assert "Config file: config.yaml" in result.stdout


def test_version_command() -> None:
    """Test version command."""
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "DeepBrief version" in result.stdout
