import json
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _get_version
from pathlib import Path
from typing import Any

import click
import torch

try:
    __version__ = _get_version("trainscope")
except PackageNotFoundError:  # pragma: no cover
    # Fallback when running from source without package metadata.
    __version__ = "1.1.0"


def _parse_skip_batches(value: str) -> list[int]:
    """Parse a comma-separated list of batch indices.

    If ``value`` starts with ``@`` the remainder is treated as a path to a text
    file that contains one or more indices separated by commas and/or
    whitespace.  Duplicate/blank entries are ignored.
    """
    text: str
    if value.startswith("@"):
        path = Path(value[1:])
        if not path.exists():
            raise click.ClickException(f"Skip batch file not found: {path}")
        text = path.read_text(encoding="utf-8")
        parts = re.split(r"[,\s]+", text.strip())
    else:
        parts = value.split(",")

    indices: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError as exc:
            raise click.ClickException(f"Invalid batch index: {part!r}") from exc
    return indices


@click.group()
@click.version_option(version=__version__, prog_name="trainscope")
def cli() -> None:
    """TrainScope command-line interface."""


@cli.command()
@click.option(
    "--run",
    default=None,
    type=click.Path(
        exists=True,
        file_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Path to a single trainscope run directory",
)
@click.option(
    "--runs",
    default=None,
    type=click.Path(
        exists=True,
        file_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Path to a root directory containing multiple run directories",
)
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7007, show_default=True, type=int)
@click.option(
    "--log-level",
    default="info",
    show_default=True,
    type=click.Choice(
        ["debug", "info", "warning", "error", "critical"],
        case_sensitive=False,
    ),
)
def ui(run: Path | None, runs: Path | None, host: str, port: int, log_level: str) -> None:
    """Start the TrainScope web UI for a run (or a directory of runs)."""
    from trainscope.ui.server import start_server

    if (run is None) == (runs is None):
        raise click.ClickException("Specify exactly one of --run or --runs")

    if run is not None:
        click.echo(f"Starting TrainScope UI for run: {run}")
        start_server(str(run.resolve()), host=host, port=port, log_level=log_level)
    else:
        assert runs is not None
        click.echo(f"Starting TrainScope UI for runs under: {runs}")
        start_server(
            str(runs.resolve()),
            host=host,
            port=port,
            log_level=log_level,
            runs_root=str(runs.resolve()),
        )
    click.echo(f"Open http://{host}:{port} in your browser")


@cli.command()
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        path_type=Path,
    ),
    help="Path to the checkpoint file",
)
@click.option(
    "--skip-batches",
    required=True,
    help="Comma-separated list of batch indices to skip, or @path/to/file",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Print instructions for resuming training with SkippingDataLoader",
)
@click.option(
    "--output",
    default="replay_config.json",
    show_default=True,
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    help="Path to write the replay config",
)
def replay(checkpoint: Path, skip_batches: str, resume: bool, output: Path) -> None:
    """Generate a replay_config.json for use with SkippingDataLoader.

    This command does NOT automatically resume training. It writes a
    replay_config.json that you pass to trainscope.replay.SkippingDataLoader in
    your training script to skip the batches that caused the loss spike.
    """
    try:
        torch.load(str(checkpoint), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise click.ClickException(f"Checkpoint unreadable with weights_only=True: {exc}") from exc

    skip_list = _parse_skip_batches(skip_batches)
    if any(s < 0 for s in skip_list):
        raise click.ClickException("Batch indices must be non-negative")

    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Batches to skip ({len(skip_list)} total): {skip_list}")

    replay_config: dict[str, Any] = {
        "checkpoint": str(checkpoint.resolve()),
        "skip_batches": skip_list,
        "total_skipped": len(skip_list),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Replay config generated by trainscope. "
            f"Skips {len(skip_list)} batch(es) when loaded into SkippingDataLoader."
        ),
    }
    with open(output, "w", encoding="utf-8") as f:
        json.dump(replay_config, f, indent=2)
    click.echo(f"Saved replay config → {output.resolve()}")

    if resume:
        click.echo(
            "\nTo resume training, use SkippingDataLoader in your script:\n\n"
            "  from trainscope.replay import SkippingDataLoader\n"
            "  import json\n\n"
            "  with open('replay_config.json') as f:\n"
            "      cfg = json.load(f)\n\n"
            "  loader = SkippingDataLoader(original_loader, cfg['skip_batches'])\n"
            "  for batch in loader:\n"
            "      loss = model(batch)\n"
            "      ..."
        )


if __name__ == "__main__":
    cli()
