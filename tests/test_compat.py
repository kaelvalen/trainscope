"""Run-format compatibility tests (Phase 3).

trainscope promises that a year-old run directory still opens in the current
UI. This module builds run directories exactly the way old versions would
have written them — legacy IPC *file* format with the *old* schema (pre-0.7
column sets, before memory fields and per-layer stats were added) and the
current v1.x append-only IPC *stream* format — and verifies the UI server
serves them without error.

This is the stability-scope guarantee in code: readers must tolerate columns
they do not know about, and old files must stay readable.
"""

import json
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi.testclient import TestClient

from trainscope.core.config import TrainScopeConfig
from trainscope.io.writer import DiskWriter
from trainscope.ui.server import create_app

# ---------------------------------------------------------------------- #
# Old-format helpers
# ---------------------------------------------------------------------- #

# The exact schemas trainscope wrote before 0.3.0 (no memory fields) and
# before 0.5.1 (no per-layer stats beyond the originals). A run written by
# v0.1.0-v0.2.x used IPC *file* format with these column sets.
LEGACY_GLOBAL_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int64()),
        pa.field("loss", pa.float64()),
        pa.field("grad_norm_before_clip", pa.float64()),
        pa.field("grad_norm_after_clip", pa.float64()),
        pa.field("lr", pa.float64()),
        pa.field("optimizer_v_norm", pa.float64()),
        pa.field("step_time_ms", pa.float64()),
        pa.field("batch_index", pa.int64()),
        pa.field("is_spike", pa.bool_()),
    ]
)

LEGACY_LAYER_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int64()),
        pa.field("grad_l2_norm", pa.float64()),
        pa.field("weight_l2_norm", pa.float64()),
        pa.field("act_mean", pa.float64()),
        pa.field("act_std", pa.float64()),
        pa.field("act_max_abs", pa.float64()),
        pa.field("act_kurtosis", pa.float64()),
        pa.field("grad_nan_inf_ratio", pa.float64()),
        pa.field("hist_counts", pa.list_(pa.float64())),
        pa.field("hist_edges", pa.list_(pa.float64())),
    ]
)


def _write_legacy_ipc_file(path: Path, schema: pa.Schema, rows: list[dict]):
    """Write rows as a legacy IPC *file* (``ipc.new_file``), as pre-0.7.0
    trainscope did before the append-only stream writer existed."""
    table = pa.Table.from_pylist(rows, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_table(table)
        writer.close()


def _make_legacy_run(run: Path) -> None:
    """Write a complete run directory the way v0.1-v0.2 trainscope did."""
    run.mkdir()

    (run / "meta.json").write_text(
        json.dumps(
            {
                "model_name": "LegacyModel",
                "model_config": {"layers": 2, "hidden": 64},
                "trainscope_config": {"run_name": run.name, "detector": {"name": "changepoint"}},
                "start_time": "2025-01-01T00:00:00",
            }
        )
    )

    global_rows = [
        {
            "step": i,
            "loss": float(i) if i != 4 else 50.0,
            "grad_norm_before_clip": 1.0 + i * 0.1,
            "grad_norm_after_clip": 1.0,
            "lr": 0.001,
            "optimizer_v_norm": 0.0,
            "step_time_ms": 1.0,
            "batch_index": i,
            "is_spike": i == 4,
        }
        for i in range(6)
    ]
    _write_legacy_ipc_file(run / "global.arrow", LEGACY_GLOBAL_SCHEMA, global_rows)

    layers_dir = run / "layers"
    layers_dir.mkdir()

    def layer_row(step: int, grad_norm: float) -> dict:
        return {
            "step": step,
            "grad_l2_norm": grad_norm,
            "weight_l2_norm": 1.0,
            "act_mean": 0.0,
            "act_std": 1.0,
            "act_max_abs": 1.0,
            "act_kurtosis": 1.0,
            "grad_nan_inf_ratio": 0.0,
            "hist_counts": [1.0, 0.0, 0.0],
            "hist_edges": [0.0, 1.0, 2.0, 3.0],
        }

    for name, base in [("layer.1", 1.0), ("layer.2", 1.0)]:
        rows = [layer_row(i, base + (5.0 if i == 4 else 0.0)) for i in range(6)]
        _write_legacy_ipc_file(
            layers_dir / f"{quote(name, safe='')}.arrow",
            LEGACY_LAYER_SCHEMA,
            rows,
        )

    spikes_dir = run / "spikes"
    spikes_dir.mkdir()
    _write_legacy_ipc_file(
        spikes_dir / "spike_step_4.arrow",
        LEGACY_GLOBAL_SCHEMA,
        global_rows[3:],
    )
    spike_layers = spikes_dir / "spike_step_4_layers"
    spike_layers.mkdir()
    _write_legacy_ipc_file(
        spike_layers / f"{quote('layer.1', safe='')}.arrow",
        LEGACY_LAYER_SCHEMA,
        [layer_row(3, 1.0), layer_row(4, 6.0), layer_row(5, 1.0)],
    )

    (run / "manifest.json").write_text(
        json.dumps({"last_step": 5, "n_global_rows": 6, "layer_files": {}})
    )


def _make_current_run(run: Path) -> None:
    """Write a run with the current v1.x writer (append-only IPC streams)."""
    config = TrainScopeConfig(run_dir=str(run.parent), run_name=run.name)
    writer = DiskWriter(run, config)

    def snap(i: int) -> dict:
        return {
            "step": i,
            "loss": float(i) if i != 4 else 50.0,
            "grad_norm_before_clip": 1.0 + i * 0.1,
            "grad_norm_after_clip": 1.0,
            "lr": 0.001,
            "optimizer_v_norm": 0.0,
            "step_time_ms": 1.0,
            "batch_index": i,
            "is_spike": i == 4,
        }

    for i in range(6):
        writer.append_global(snap(i))
    # The scope persists spike windows separately; mirror it so the spike
    # appears in the UI like a real run would.
    window = [{"global": snap(i), "layers": {}, "step_number": i} for i in range(3, 6)]
    writer.write_spike_window(4, window, {})
    writer.flush()
    writer.close()


# ---------------------------------------------------------------------- #
# Legacy IPC file-format run (pre-0.7.0 writer)
# ---------------------------------------------------------------------- #
class TestLegacyFileFormatRun:
    def test_legacy_run_serves_every_view(self, tmp_path):
        run = tmp_path / "legacy_run"
        _make_legacy_run(run)
        client = TestClient(create_app(str(run)))

        assert client.get("/api/manifest").status_code == 200
        assert client.get("/api/meta").status_code == 200

        r = client.get("/api/global")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 6
        # Old schema has no memory/spike_score columns; the reader must not
        # synthesize them or fail — it returns exactly what the file holds.
        assert "cpu_memory_mb" not in rows[0]
        assert rows[4]["is_spike"] is True

        r = client.get("/api/layers")
        assert r.status_code == 200
        assert sorted(r.json()) == ["layer.1", "layer.2"]

        r = client.get("/api/layers/ranked?top_n=10")
        assert r.status_code == 200
        assert r.json() == ["layer.1", "layer.2"]

        r = client.get("/api/layers/layer.1")
        assert r.status_code == 200
        assert len(r.json()) == 6

        r = client.get("/api/spikes")
        assert r.status_code == 200
        assert r.json() == [{"step": 4, "file": "spike_step_4.arrow"}]

        r = client.get("/api/spikes/4")
        assert r.status_code == 200
        assert len(r.json()) == 3

        r = client.get("/api/spikes/4/layers")
        assert r.status_code == 200
        assert r.json() == ["layer.1"]

        r = client.get("/api/spikes/4/layers/layer.1")
        assert r.status_code == 200
        assert len(r.json()) == 3

    def test_legacy_run_diff_and_cluster(self, tmp_path):
        run = tmp_path / "legacy_run"
        _make_legacy_run(run)
        client = TestClient(create_app(str(run)))

        # Diff needs hist_counts on both steps — the legacy layer file has them.
        r = client.get("/api/diff?step_a=2&step_b=4")
        assert r.status_code == 200
        data = r.json()
        assert data and data[0]["layer"] == "layer.1"

        # Multi-run mode must discover and summarise the legacy run.
        root = tmp_path
        multi = TestClient(create_app(str(run), runs_root=str(root)))
        r = multi.get("/api/runs")
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert r.json()[0]["name"] == "legacy_run"
        assert r.json()[0]["spike_count"] == 1
        assert r.json()[0]["last_loss"] == 5.0


# ---------------------------------------------------------------------- #
# Current v1.x append-only stream-format run
# ---------------------------------------------------------------------- #
class TestCurrentStreamFormatRun:
    def test_current_run_serves_every_view(self, tmp_path):
        run = tmp_path / "current_run"
        _make_current_run(run)
        client = TestClient(create_app(str(run)))

        r = client.get("/api/global")
        assert r.status_code == 200
        rows = r.json()
        assert len(rows) == 6
        assert rows[4]["is_spike"] is True

        r = client.get("/api/spikes")
        assert r.status_code == 200
        assert r.json() == [{"step": 4, "file": "spike_step_4.arrow"}]

        assert client.get("/api/manifest").status_code == 200
        assert client.get("/api/meta").status_code == 200

    def test_mixed_run_same_root_multi_run(self, tmp_path):
        """A root containing a legacy file-format run AND a current
        stream-format run must list both and switch between them."""
        legacy = tmp_path / "legacy_run"
        _make_legacy_run(legacy)
        current = tmp_path / "current_run"
        _make_current_run(current)

        client = TestClient(create_app(str(legacy), runs_root=str(tmp_path)))
        r = client.get("/api/runs")
        assert r.status_code == 200
        assert {row["name"] for row in r.json()} == {"legacy_run", "current_run"}

        r = client.post("/api/runs/select", json={"name": "current_run"})
        assert r.status_code == 200
        assert r.json()["name"] == "current_run"
        assert client.get("/api/global").json()[-1]["step"] == 5

        r = client.post("/api/runs/select", json={"name": "legacy_run"})
        assert r.status_code == 200
        assert r.json()["name"] == "legacy_run"
        assert client.get("/api/global").json()[-1]["step"] == 5
