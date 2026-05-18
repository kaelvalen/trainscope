import json
from pathlib import Path

import click


@click.group()
def cli():
    pass


@cli.command()
@click.option("--run", required=True, help="Path to the trainscope run directory")
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=7007, show_default=True, type=int)
def ui(run: str, host: str, port: int):
    from trainscope.ui.server import start_server
    click.echo(f"Starting TrainScope UI for run: {run}")
    click.echo(f"Open http://{host}:{port} in your browser")
    start_server(run, host=host, port=port)


@cli.command()
@click.option("--checkpoint", required=True, help="Path to the checkpoint file")
@click.option(
    "--skip-batches",
    required=True,
    help="Comma-separated list of batch indices to skip",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Print instructions for resuming training with SkippingDataLoader",
)
def replay(checkpoint: str, skip_batches: str, resume: bool):
    """Generate a replay_config.json for use with SkippingDataLoader.

    This command does NOT automatically resume training. It writes a
    replay_config.json that you pass to trainscope.replay.SkippingDataLoader in
    your training script to skip the batches that caused the loss spike.
    """
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise click.ClickException(f"Checkpoint not found: {checkpoint}")

    import torch
    torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    skip_list = [int(b.strip()) for b in skip_batches.split(",") if b.strip()]

    click.echo(f"Checkpoint: {checkpoint}")
    click.echo(f"Batches to skip ({len(skip_list)} total): {skip_list}")

    replay_config = {
        "checkpoint": str(ckpt_path.resolve()),
        "skip_batches": skip_list,
    }
    out_path = Path("replay_config.json")
    with open(out_path, "w") as f:
        json.dump(replay_config, f, indent=2)
    click.echo(f"Saved replay config → {out_path.resolve()}")

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
