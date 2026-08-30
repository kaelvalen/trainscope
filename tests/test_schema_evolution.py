"""Arrow schema-evolution tests (Phase 3).

The stability scope promises: Arrow files are additive-only within a major
version. Writers may add columns; readers must tolerate columns they do not
know about, and files written by older versions (with *fewer* columns) must
stay readable.

These tests pin that contract: a file whose schema is a superset of today's
(extra nullable columns) and a file whose schema is a strict subset (old
columns missing) both round-trip through ``read_arrow_rows_sync`` and through
the UI server without error or data loss.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from fastapi.testclient import TestClient

from trainscope.io.writer import (
    GLOBAL_SCHEMA,
    LAYER_SCHEMA,
    read_arrow_rows_sync,
)
from trainscope.ui.server import create_app


def _write_arrow(path: Path, schema: pa.Schema, rows: list[dict]):
    table = pa.Table.from_pylist(rows, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_table(table)
        writer.close()


def _minimal_run(run: Path, global_schema: pa.Schema, layer_schema: pa.Schema) -> None:
    run.mkdir()
    (run / "meta.json").write_text(
        json.dumps(
            {
                "model_name": "EvolModel",
                "model_config": {},
                "trainscope_config": {"run_name": run.name},
            }
        )
    )
    (run / "manifest.json").write_text(
        json.dumps({"last_step": 1, "n_global_rows": 2, "layer_files": {}})
    )


class TestReaderToleratesUnknownColumns:
    def test_extra_global_column_round_trips(self, tmp_path):
        """A future minor release may add a nullable column to global.arrow;
        the current reader must return it instead of failing or dropping it."""
        extra_schema = GLOBAL_SCHEMA.append(pa.field("token_entropy", pa.float64()))
        rows = [
            {
                "step": 0,
                "loss": 1.0,
                "grad_norm_before_clip": 0.5,
                "grad_norm_after_clip": 0.5,
                "lr": 0.001,
                "optimizer_v_norm": 0.0,
                "step_time_ms": 1.0,
                "batch_index": 0,
                "is_spike": False,
                "cpu_memory_mb": 0.0,
                "cuda_memory_mb": 0.0,
                "spike_score": 0.0,
                "token_entropy": 6.3,
            }
        ]
        path = tmp_path / "global.arrow"
        _write_arrow(path, extra_schema, rows)

        out = read_arrow_rows_sync(path)
        assert len(out) == 1
        assert out[0]["token_entropy"] == 6.3
        assert out[0]["loss"] == 1.0

    def test_extra_global_column_served_by_ui(self, tmp_path):
        """The UI must keep serving a run whose global.arrow has columns the
        server does not know about."""
        extra_schema = GLOBAL_SCHEMA.append(pa.field("token_entropy", pa.float64()))
        run = tmp_path / "run"
        _minimal_run(run, extra_schema, LAYER_SCHEMA)
        _write_arrow(
            run / "global.arrow",
            extra_schema,
            [
                {
                    "step": 0,
                    "loss": 1.0,
                    "grad_norm_before_clip": 0.5,
                    "grad_norm_after_clip": 0.5,
                    "lr": 0.001,
                    "optimizer_v_norm": 0.0,
                    "step_time_ms": 1.0,
                    "batch_index": 0,
                    "is_spike": False,
                    "cpu_memory_mb": 0.0,
                    "cuda_memory_mb": 0.0,
                    "spike_score": 0.0,
                    "token_entropy": 6.3,
                }
            ],
        )

        client = TestClient(create_app(str(run)))
        r = client.get("/api/global")
        assert r.status_code == 200
        assert r.json()[0]["token_entropy"] == 6.3

    def test_extra_layer_column_round_trips(self, tmp_path):
        extra_schema = LAYER_SCHEMA.append(pa.field("quantile_99", pa.float64()))
        rows = [
            {
                "step": 0,
                "grad_l2_norm": 0.3,
                "weight_l2_norm": 1.2,
                "act_mean": 0.0,
                "act_std": 1.0,
                "act_max_abs": 3.5,
                "act_kurtosis": 0.1,
                "grad_nan_inf_ratio": 0.0,
                "hist_counts": [1.0, 0.0],
                "hist_edges": [0.0, 1.0, 2.0],
                "grad_max_abs": 0.3,
                "grad_mean": 0.0,
                "weight_mean": 0.0,
                "weight_std": 1.0,
                "weight_max_abs": 1.2,
                "weight_min": -1.2,
                "act_min": -3.5,
                "act_max": 3.5,
                "act_median": 0.0,
                "quantile_99": 2.9,
            }
        ]
        path = tmp_path / "layer.arrow"
        _write_arrow(path, extra_schema, rows)

        out = read_arrow_rows_sync(path)
        assert out[0]["quantile_99"] == 2.9
        assert out[0]["grad_l2_norm"] == 0.3


class TestReaderToleratesOlderSubset:
    """Files written by old versions have fewer columns than today's schema.
    Readers must return the rows that exist (no synthesized zeros) and the UI
    must not crash on the missing keys."""

    def test_global_without_later_columns(self, tmp_path):
        """Pre-1.0 global.arrow lacked nothing structural after 0.5.1, but an
        even older file (pre-0.3.0) has no cpu/cuda memory columns at all."""
        old_schema = pa.schema(
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
        rows = [
            {
                "step": 0,
                "loss": 1.0,
                "grad_norm_before_clip": 0.5,
                "grad_norm_after_clip": 0.5,
                "lr": 0.001,
                "optimizer_v_norm": 0.0,
                "step_time_ms": 1.0,
                "batch_index": 0,
                "is_spike": False,
            }
        ]
        path = tmp_path / "global.arrow"
        _write_arrow(path, old_schema, rows)

        out = read_arrow_rows_sync(path)
        assert len(out) == 1
        assert out[0]["loss"] == 1.0
        assert out[0]["is_spike"] is False
        # Missing columns must be absent, not silently zero-filled: a fake 0.0
        # memory reading would mislead the UI into rendering data that was
        # never recorded.
        assert "cpu_memory_mb" not in out[0]
        assert "spike_score" not in out[0]

    def test_old_global_served_by_ui(self, tmp_path):
        old_schema = pa.schema(
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
        run = tmp_path / "run"
        _minimal_run(run, old_schema, LAYER_SCHEMA)
        _write_arrow(
            run / "global.arrow",
            old_schema,
            [
                {
                    "step": 0,
                    "loss": 1.0,
                    "grad_norm_before_clip": 0.5,
                    "grad_norm_after_clip": 0.5,
                    "lr": 0.001,
                    "optimizer_v_norm": 0.0,
                    "step_time_ms": 1.0,
                    "batch_index": 0,
                    "is_spike": False,
                }
            ],
        )

        client = TestClient(create_app(str(run)))
        r = client.get("/api/global")
        assert r.status_code == 200
        row = r.json()[0]
        assert row["loss"] == 1.0
        assert "cpu_memory_mb" not in row

    def test_layer_without_later_columns(self, tmp_path):
        """Pre-0.5.1 layer files lacked the weight/grad summary columns."""
        old_schema = pa.schema(
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
        rows = [
            {
                "step": 0,
                "grad_l2_norm": 0.3,
                "weight_l2_norm": 1.2,
                "act_mean": 0.0,
                "act_std": 1.0,
                "act_max_abs": 3.5,
                "act_kurtosis": 0.1,
                "grad_nan_inf_ratio": 0.0,
                "hist_counts": [1.0, 0.0],
                "hist_edges": [0.0, 1.0, 2.0],
            }
        ]
        path = tmp_path / "layer.arrow"
        _write_arrow(path, old_schema, rows)

        out = read_arrow_rows_sync(path)
        assert out[0]["grad_l2_norm"] == 0.3
        assert "weight_mean" not in out[0]
        assert "act_median" not in out[0]

    def test_old_layer_served_by_ui_ranked_and_diff(self, tmp_path):
        """Ranked-layer variance and diff must work on old layer files even
        though the newer per-layer columns are missing."""
        from urllib.parse import quote

        old_schema = pa.schema(
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
        run = tmp_path / "run"
        run.mkdir()
        (run / "meta.json").write_text(
            json.dumps(
                {
                    "model_name": "EvolModel",
                    "model_config": {},
                    "trainscope_config": {"run_name": run.name},
                }
            )
        )
        (run / "manifest.json").write_text(
            json.dumps({"last_step": 1, "n_global_rows": 2, "layer_files": {}})
        )
        layers_dir = run / "layers"
        layers_dir.mkdir()

        def row(step: int, grad_norm: float) -> dict:
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

        _write_arrow(
            layers_dir / f"{quote('layer.a', safe='')}.arrow",
            old_schema,
            [row(0, 1.0), row(1, 2.0)],
        )
        _write_arrow(
            layers_dir / f"{quote('layer.b', safe='')}.arrow",
            old_schema,
            [row(0, 1.0), row(1, 1.1)],
        )

        client = TestClient(create_app(str(run)))
        r = client.get("/api/layers/ranked?top_n=2")
        assert r.status_code == 200
        assert r.json() == ["layer.a", "layer.b"]

        r = client.get("/api/diff?step_a=0&step_b=1")
        assert r.status_code == 200
        assert len(r.json()) == 2
        assert {item["layer"] for item in r.json()} == {"layer.a", "layer.b"}
