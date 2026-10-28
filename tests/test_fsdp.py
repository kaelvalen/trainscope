"""TrainScope under FSDP2 (``fully_shard``): two CPU processes over gloo."""

import copy
import datetime
import math
from typing import Any

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

fsdp = pytest.importorskip("torch.distributed.fsdp")
if not dist.is_available() or not hasattr(fsdp, "fully_shard"):
    pytest.skip("FSDP2 (fully_shard) not available", allow_module_level=True)

WORLD = 2
STEPS = 4


def _grad_norm(p: torch.Tensor) -> float:
    assert p.grad is not None
    return float(p.grad.norm())


def _worker(rank: int, init_file: str, run_dir: str) -> None:
    from trainscope import TrainScope, TrainScopeConfig
    from trainscope.io.writer import read_arrow_rows_sync

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=WORLD,
        timeout=datetime.timedelta(seconds=60),
    )
    torch.manual_seed(0)
    # The last bias has 1 element, so rank 1's shard of it is empty.
    model = nn.Sequential(nn.Linear(6, 5), nn.Tanh(), nn.Linear(5, 1))
    reference = copy.deepcopy(model)
    for layer in (model[0], model[2]):
        fsdp.fully_shard(layer)
    fsdp.fully_shard(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    ref_opt = torch.optim.AdamW(reference.parameters(), lr=1e-2)

    # Rank 0 flags every step after warmup as a spike, rank 1 never does: the
    # spike-triggered histograms are collectives and must not deadlock.
    detector = (
        {"name": "z_score", "threshold": -1.0, "min_observations": 1}
        if rank == 0
        else {"name": "z_score", "threshold": 1e9}
    )
    cfg = TrainScopeConfig(
        run_dir=run_dir,
        run_name="fsdp",
        rank=rank,
        track_memory=False,
        histogram_every_n_steps=1000,
        detector=detector,
    )
    scope = TrainScope(model, opt, cfg).attach()

    expected: list[dict[str, Any]] = []
    for step in range(STEPS):
        gen = torch.Generator().manual_seed(step)  # same batch on both ranks
        x, y = torch.randn(4, 6, generator=gen), torch.randn(4, 1, generator=gen)
        for m, o in ((model, opt), (reference, ref_opt)):
            o.zero_grad()
            nn.functional.mse_loss(m(x), y).backward()
        expected.append(
            {
                "grad_norm": math.sqrt(sum(_grad_norm(p) ** 2 for p in reference.parameters())),
                "layers": {
                    n: (_grad_norm(p), float(p.norm()), float(p.std(unbiased=False)))
                    for n, p in reference.named_parameters()
                },
            }
        )
        scope.step(float(nn.functional.mse_loss(model(x), y)))
        opt.step()
        ref_opt.step()
    scope.detach()

    from pathlib import Path

    run = Path(run_dir) / f"fsdp_rank{rank}"
    rows = read_arrow_rows_sync(run / "global.arrow")
    for row, exp in zip(rows, expected):
        assert row["grad_norm_before_clip"] == pytest.approx(exp["grad_norm"], rel=1e-5)
    for name, (g, w, s) in expected[-1]["layers"].items():
        layer_rows = read_arrow_rows_sync(run / "layers" / f"{name}.arrow")
        last = layer_rows[-1]
        assert last["grad_l2_norm"] == pytest.approx(g, rel=1e-5), name
        assert last["weight_l2_norm"] == pytest.approx(w, rel=1e-5), name
        assert last["weight_std"] == pytest.approx(s, rel=1e-4, abs=1e-7), name
        spiked = [r for r in layer_rows if r["hist_counts"]]
        # Both ranks histogram on rank 0's spikes; counts cover the whole tensor.
        assert spiked, name
        numel = reference.get_parameter(name).numel()
        assert all(sum(r["hist_counts"]) == numel for r in spiked), name
    dist.destroy_process_group()


def test_trainscope_records_global_metrics_under_fsdp2(tmp_path):
    mp.spawn(_worker, args=(str(tmp_path / "init"), str(tmp_path)), nprocs=WORLD)
