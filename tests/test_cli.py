"""Tests for trainscope.cli."""

import json
from pathlib import Path
from unittest.mock import patch

import click.testing
import pytest
import torch
import torch.nn as nn

from trainscope.cli import __version__, cli


@pytest.fixture
def runner():
    return click.testing.CliRunner()


@pytest.fixture
def checkpoint(tmp_path: Path):
    """Create a tiny checkpoint file that torch.load can read."""
    model = nn.Linear(4, 1)
    path = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict(), "step": 42}, path)
    return path


def test_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert f"trainscope, version {__version__}" in result.output


def test_ui_rejects_missing_run(runner):
    result = runner.invoke(cli, ["ui", "--run", "/nonexistent/path"])
    assert result.exit_code != 0
    assert "does not exist" in result.output or "Invalid value" in result.output


def test_ui_starts_server(runner, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch("trainscope.ui.server.start_server") as mock_start:
        result = runner.invoke(cli, ["ui", "--run", str(run_dir)])
    assert result.exit_code == 0, result.output
    mock_start.assert_called_once()
    args, kwargs = mock_start.call_args
    assert str(run_dir.resolve()) in args or kwargs.get("run_path") == str(run_dir.resolve())


def test_ui_passes_host_port_log_level(runner, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch("trainscope.ui.server.start_server") as mock_start:
        result = runner.invoke(
            cli,
            [
                "ui",
                "--run",
                str(run_dir),
                "--host",
                "0.0.0.0",
                "--port",
                "9000",
                "--log-level",
                "debug",
            ],
        )
    assert result.exit_code == 0, result.output
    mock_start.assert_called_once_with(
        str(run_dir.resolve()), host="0.0.0.0", port=9000, log_level="debug"
    )


def test_ui_requires_exactly_one_of_run_or_runs(runner, tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with patch("trainscope.ui.server.start_server") as mock_start:
        # Both provided -> rejected.
        result = runner.invoke(cli, ["ui", "--run", str(run_dir), "--runs", str(run_dir)])
    assert result.exit_code != 0
    assert "exactly one" in result.output
    mock_start.assert_not_called()

    # Neither provided -> rejected.
    with patch("trainscope.ui.server.start_server") as mock_start:
        result = runner.invoke(cli, ["ui"])
    assert result.exit_code != 0
    assert "exactly one" in result.output
    mock_start.assert_not_called()


def test_ui_starts_server_with_runs_root(runner, tmp_path: Path):
    root = tmp_path / "runs_root"
    root.mkdir()
    with patch("trainscope.ui.server.start_server") as mock_start:
        result = runner.invoke(cli, ["ui", "--runs", str(root)])
    assert result.exit_code == 0, result.output
    args, kwargs = mock_start.call_args
    assert args[0] == str(root.resolve())
    assert kwargs.get("runs_root") == str(root.resolve())


def test_replay_writes_config(runner, checkpoint: Path, tmp_path: Path):
    output = tmp_path / "replay_config.json"
    result = runner.invoke(
        cli,
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--skip-batches",
            "1,3,5",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()
    cfg = json.loads(output.read_text())
    assert cfg["checkpoint"] == str(checkpoint.resolve())
    assert cfg["skip_batches"] == [1, 3, 5]
    assert cfg["total_skipped"] == 3
    assert "generated_at" in cfg


def test_replay_from_file(runner, checkpoint: Path, tmp_path: Path):
    skip_file = tmp_path / "skip.txt"
    skip_file.write_text(" 2 , 4 \n 6 ")
    output = tmp_path / "replay_config.json"
    result = runner.invoke(
        cli,
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--skip-batches",
            f"@{skip_file}",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    cfg = json.loads(output.read_text())
    assert cfg["skip_batches"] == [2, 4, 6]


def test_replay_resume_flag(runner, checkpoint: Path, tmp_path: Path):
    output = tmp_path / "replay_config.json"
    result = runner.invoke(
        cli,
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--skip-batches",
            "0",
            "--resume",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "SkippingDataLoader" in result.output


def test_replay_rejects_negative_batch(runner, checkpoint: Path, tmp_path: Path):
    output = tmp_path / "replay_config.json"
    result = runner.invoke(
        cli,
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--skip-batches",
            "1,-1,3",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "non-negative" in result.output


def test_replay_rejects_invalid_batch(runner, checkpoint: Path, tmp_path: Path):
    output = tmp_path / "replay_config.json"
    result = runner.invoke(
        cli,
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--skip-batches",
            "1,foo,3",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert "Invalid batch index" in result.output


def test_replay_rejects_missing_skip_file(runner, checkpoint: Path):
    result = runner.invoke(
        cli,
        [
            "replay",
            "--checkpoint",
            str(checkpoint),
            "--skip-batches",
            "@/nonexistent/skip.txt",
        ],
    )
    assert result.exit_code != 0
    assert "Skip batch file not found" in result.output


def test_replay_rejects_unreadable_checkpoint(runner, tmp_path: Path):
    bad = tmp_path / "bad.pt"
    bad.write_text("not a checkpoint")
    result = runner.invoke(
        cli,
        ["replay", "--checkpoint", str(bad), "--skip-batches", "1"],
    )
    assert result.exit_code != 0
    assert "weights_only" in result.output
