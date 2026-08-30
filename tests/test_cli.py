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


# --------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------- #
def _make_report_run(path: Path, name: str, loss_spike: bool = False) -> None:
    """Minimal run directory: meta.json + global.arrow + optional spike."""
    import pyarrow as pa
    import pyarrow.ipc as ipc

    from trainscope.io.writer import GLOBAL_SCHEMA

    path.mkdir(parents=True, exist_ok=True)
    (path / "meta.json").write_text(
        json.dumps(
            {
                "model_name": "MiniGPT",
                "model_config": {"layers": 2, "hidden": 64},
                "trainscope_config": {
                    "run_name": name,
                    "detector": {"name": "changepoint"},
                },
                "start_time": "2026-01-01T00:00:00",
            }
        )
    )
    rows = []
    for i in range(80):
        loss = 100.0 if (loss_spike and i == 60) else float(i)
        rows.append(
            {
                "step": i,
                "loss": loss,
                "grad_norm_before_clip": 1.0,
                "grad_norm_after_clip": 1.0,
                "lr": 0.001,
                "optimizer_v_norm": 0.0,
                "step_time_ms": 1.0,
                "batch_index": i,
                "is_spike": loss_spike and i == 60,
            }
        )
    table = pa.Table.from_pylist(rows, schema=GLOBAL_SCHEMA)
    with pa.OSFile(str(path / "global.arrow"), "wb") as sink:
        writer = ipc.new_file(sink, GLOBAL_SCHEMA)
        writer.write_table(table)
        writer.close()
    if loss_spike:
        spikes = path / "spikes"
        spikes.mkdir()
        with pa.OSFile(str(spikes / "spike_step_60.arrow"), "wb") as sink:
            writer = ipc.new_file(sink, GLOBAL_SCHEMA)
            writer.write_table(table.slice(60, 1))
            writer.close()


def test_report_run_markdown(runner, tmp_path: Path):
    run_dir = tmp_path / "report_run"
    _make_report_run(run_dir, "report_run", loss_spike=True)

    result = runner.invoke(cli, ["report", "--run", str(run_dir)])
    assert result.exit_code == 0, result.output
    assert "# Post-mortem: report_run" in result.output
    assert "- Model: MiniGPT" in result.output
    assert "## Signal signature" in result.output
    assert "## Spikes" in result.output
    assert "Spike steps: 60" in result.output
    assert result.output.count("##") > 0


def test_report_run_json(runner, tmp_path: Path):
    run_dir = tmp_path / "report_run"
    _make_report_run(run_dir, "report_run", loss_spike=True)

    result = runner.invoke(cli, ["report", "--run", str(run_dir), "--format", "json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["run"] == "report_run"
    assert data["config"]["model_name"] == "MiniGPT"
    assert data["signal_signature"]["first"] == "loss"
    assert data["spike_steps"] == [60]


def test_report_run_to_file(runner, tmp_path: Path):
    run_dir = tmp_path / "report_run"
    _make_report_run(run_dir, "report_run")
    output = tmp_path / "report.md"

    result = runner.invoke(cli, ["report", "--run", str(run_dir), "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert output.exists()
    assert "Post-mortem: report_run" in output.read_text()


def test_report_clusters(runner, tmp_path: Path):
    root = tmp_path / "runs"
    root.mkdir()
    _make_report_run(root / "spiked", "spiked", loss_spike=True)
    _make_report_run(root / "calm", "calm")

    result = runner.invoke(cli, ["report", "--runs", str(root)])
    assert result.exit_code == 0, result.output
    assert "# Run behavior clusters (2 runs)" in result.output
    assert "## loss-led (1 runs)" in result.output
    assert "## Stable (no signal) runs" in result.output
    assert "- calm" in result.output


def test_report_requires_exactly_one_of_run_or_runs(runner, tmp_path: Path):
    run_dir = tmp_path / "r"
    _make_report_run(run_dir, "r")

    result = runner.invoke(cli, ["report"])
    assert result.exit_code != 0
    assert "exactly one" in result.output

    result = runner.invoke(cli, ["report", "--run", str(run_dir), "--runs", str(run_dir)])
    assert result.exit_code != 0
    assert "exactly one" in result.output
