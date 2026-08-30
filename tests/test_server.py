import json
from pathlib import Path
from urllib.parse import quote

import pyarrow as pa
import pyarrow.ipc as ipc
import pytest
from fastapi.testclient import TestClient

from trainscope.io.writer import GLOBAL_SCHEMA, LAYER_SCHEMA
from trainscope.ui.server import create_app


def _write_arrow(path, schema, rows):
    table = pa.Table.from_pylist(rows, schema=schema)
    with pa.OSFile(str(path), "wb") as sink:
        writer = ipc.new_file(sink, schema)
        writer.write_table(table)
        writer.close()


@pytest.fixture
def run_dir(tmp_path):
    run = tmp_path / "run"
    run.mkdir()

    meta = {
        "model_name": "TestModel",
        "model_config": {},
        "trainscope_config": {"run_name": "test-run"},
    }
    (run / "meta.json").write_text(json.dumps(meta))

    global_rows = [
        {
            "step": i,
            "loss": float(i),
            "grad_norm_before_clip": 1.0 + i * 0.1,
            "grad_norm_after_clip": 1.0,
            "lr": 0.001,
            "optimizer_v_norm": 0.0,
            "step_time_ms": 1.0,
            "batch_index": i,
            "is_spike": i == 3,
            "cpu_memory_mb": 0.0,
            "cuda_memory_mb": 0.0,
        }
        for i in range(5)
    ]
    _write_arrow(run / "global.arrow", GLOBAL_SCHEMA, global_rows)

    layers_dir = run / "layers"
    layers_dir.mkdir()

    def layer_row(step, grad_norm, hist_counts):
        return {
            "step": step,
            "grad_l2_norm": grad_norm,
            "weight_l2_norm": 1.0,
            "act_mean": 0.0,
            "act_std": 1.0,
            "act_max_abs": 1.0,
            "act_kurtosis": 1.0,
            "grad_nan_inf_ratio": 0.0,
            "hist_counts": hist_counts,
            "hist_edges": list(range(len(hist_counts) + 1)),
            "grad_max_abs": grad_norm,
            "grad_mean": 0.0,
            "weight_mean": 0.0,
            "weight_std": 1.0,
            "weight_max_abs": 1.0,
            "weight_min": -1.0,
            "act_min": -1.0,
            "act_max": 1.0,
            "act_median": 0.0,
        }

    layer_1_rows = [
        layer_row(0, 1.0, [1.0, 0.0, 0.0]),
        layer_row(1, 1.1, [0.0, 1.0, 0.0]),
        layer_row(2, 1.2, [0.0, 0.0, 1.0]),
        layer_row(3, 10.0, [1.0, 1.0, 1.0]),
        layer_row(4, 1.3, [1.0, 0.0, 0.0]),
    ]
    _write_arrow(
        layers_dir / f"{quote('layer.1', safe='')}.arrow",
        LAYER_SCHEMA,
        layer_1_rows,
    )

    layer_2_rows = [
        layer_row(0, 1.0, [1.0, 0.0]),
        layer_row(1, 1.0, [1.0, 0.0]),
        layer_row(2, 1.0, [1.0, 0.0]),
        layer_row(3, 1.0, [1.0, 0.0]),
        layer_row(4, 1.0, [1.0, 0.0]),
    ]
    _write_arrow(
        layers_dir / f"{quote('transformer/h/0', safe='')}.arrow",
        LAYER_SCHEMA,
        layer_2_rows,
    )

    spikes_dir = run / "spikes"
    spikes_dir.mkdir()
    spike_global = [global_rows[2], global_rows[3], global_rows[4]]
    _write_arrow(spikes_dir / "spike_step_3.arrow", GLOBAL_SCHEMA, spike_global)

    spike_layers_dir = spikes_dir / "spike_step_3_layers"
    spike_layers_dir.mkdir()
    _write_arrow(
        spike_layers_dir / f"{quote('layer.1', safe='')}.arrow",
        LAYER_SCHEMA,
        layer_1_rows[2:5],
    )

    manifest = {
        "last_step": 4,
        "n_global_rows": 5,
        "layer_files": {},
    }
    (run / "manifest.json").write_text(json.dumps(manifest))

    return run


@pytest.fixture
def client(run_dir):
    return TestClient(create_app(str(run_dir)))


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["exists"] is True
    static_index = Path(__file__).parent.parent / "trainscope" / "ui" / "static" / "index.html"
    assert data["static_served"] is static_index.exists()


def test_manifest(client):
    r = client.get("/api/manifest")
    assert r.status_code == 200
    assert r.json()["last_step"] == 4


def test_meta(client, run_dir):
    r = client.get("/api/meta")
    assert r.status_code == 200
    assert r.json()["model_name"] == "TestModel"


def test_meta_missing(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    c = TestClient(create_app(str(empty)))
    r = c.get("/api/meta")
    assert r.status_code == 200
    assert r.json()["model_name"] == "Training in progress..."


def test_global(client):
    r = client.get("/api/global")
    assert r.status_code == 200
    assert len(r.json()) == 5
    assert r.json()[3]["is_spike"] is True


def test_replay_no_config(client):
    """Without a replay_config.json the endpoint reports an empty plan."""
    r = client.get("/api/replay")
    assert r.status_code == 200
    data = r.json()
    assert data["config"] is None
    assert data["skipped_batches"] == []
    assert data["skipped_steps"] == []
    assert data["n_global_steps"] == 5


def test_replay_maps_skipped_batches_to_steps(run_dir):
    """A replay config's skipped batch indices are mapped onto the training
    steps that consumed those batches via global.arrow's batch_index."""
    (run_dir / "replay_config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(run_dir / "checkpoints" / "4.pt"),
                "skip_batches": [2, 3],
                "total_skipped": 2,
            }
        )
    )
    client = TestClient(create_app(str(run_dir)))
    r = client.get("/api/replay")
    assert r.status_code == 200
    data = r.json()
    assert data["config"]["checkpoint"].endswith("checkpoints/4.pt")
    assert data["skipped_batches"] == [2, 3]
    # batch_index == step in this fixture, so the skipped steps are 2 and 3.
    assert data["skipped_steps"] == [2, 3]
    assert data["n_global_steps"] == 5


def test_replay_skips_unrecorded_batch_indices(run_dir):
    """Batch indices that no global row consumed (e.g. out of range) must not
    produce phantom skipped steps."""
    (run_dir / "replay_config.json").write_text(
        json.dumps({"skip_batches": [99, 100], "total_skipped": 2})
    )
    client = TestClient(create_app(str(run_dir)))
    data = client.get("/api/replay").json()
    assert data["skipped_batches"] == [99, 100]
    assert data["skipped_steps"] == []


def test_websocket_streams_global_rows_after_empty_start(tmp_path):
    run = tmp_path / "live"
    run.mkdir()

    row = {
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
    }

    with TestClient(create_app(str(run))) as live_client:
        with live_client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json()["type"] == "meta"
            _write_arrow(run / "global.arrow", GLOBAL_SCHEMA, [row])

            message = websocket.receive_json()
            assert message["type"] == "global"
            assert message["payload"] == [row]


def test_websocket_streams_only_new_global_rows(tmp_path):
    run = tmp_path / "live"
    run.mkdir()

    def row(step):
        return {
            "step": step,
            "loss": 1.0 + step,
            "grad_norm_before_clip": 0.5,
            "grad_norm_after_clip": 0.5,
            "lr": 0.001,
            "optimizer_v_norm": 0.0,
            "step_time_ms": 1.0,
            "batch_index": step,
            "is_spike": False,
            "cpu_memory_mb": 0.0,
            "cuda_memory_mb": 0.0,
        }

    with TestClient(create_app(str(run))) as live_client:
        with live_client.websocket_connect("/ws") as websocket:
            assert websocket.receive_json()["type"] == "meta"
            _write_arrow(run / "global.arrow", GLOBAL_SCHEMA, [row(0), row(1)])
            initial = websocket.receive_json()
            assert initial["type"] == "global"
            assert [item["step"] for item in initial["payload"]] == [0, 1]

            _write_arrow(run / "global.arrow", GLOBAL_SCHEMA, [row(0), row(1), row(2)])
            delta = websocket.receive_json()
            assert delta["type"] == "global_delta"
            assert [item["step"] for item in delta["payload"]] == [2]


def test_layers(client):
    r = client.get("/api/layers")
    assert r.status_code == 200
    assert r.json() == ["layer.1", "transformer/h/0"]


def test_layer(client):
    r = client.get("/api/layers/layer.1")
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_layer_with_slash(client):
    r = client.get("/api/layers/transformer/h/0")
    assert r.status_code == 200
    assert len(r.json()) == 5


def test_layer_not_found(client):
    r = client.get("/api/layers/missing")
    assert r.status_code == 404


def test_spikes(client):
    r = client.get("/api/spikes")
    assert r.status_code == 200
    assert r.json() == [{"step": 3, "file": "spike_step_3.arrow"}]


def test_spike(client):
    r = client.get("/api/spikes/3")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_spike_layers(client):
    r = client.get("/api/spikes/3/layers")
    assert r.status_code == 200
    assert r.json() == ["layer.1"]


def test_spike_layer(client):
    r = client.get("/api/spikes/3/layers/layer.1")
    assert r.status_code == 200
    assert len(r.json()) == 3


def test_diff(client):
    r = client.get("/api/diff?step_a=0&step_b=2")
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 2
    layers = {item["layer"] for item in data}
    assert "layer.1" in layers
    assert "transformer/h/0" in layers
    # layer.1 had different histograms, so KL should be > 0.
    layer_1_kl = next(item["kl_divergence"] for item in data if item["layer"] == "layer.1")
    assert layer_1_kl > 0.0


def test_diff_invalid_step(client):
    r = client.get("/api/diff?step_a=-1&step_b=2")
    assert r.status_code == 422


def test_layers_ranked(client):
    r = client.get("/api/layers/ranked?top_n=1")
    assert r.status_code == 200
    assert r.json() == ["layer.1"]


def test_layers_ranked_default(client):
    r = client.get("/api/layers/ranked")
    assert r.status_code == 200
    assert len(r.json()) == 2


def test_layers_ranked_invalid_top_n(client):
    r = client.get("/api/layers/ranked?top_n=0")
    assert r.status_code == 422


def test_fallback_root(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    c = TestClient(create_app(str(empty)))
    r = c.get("/")
    assert r.status_code == 200
    assert "TrainScope" in r.text


def test_cors_headers(client):
    r = client.get("/api/health", headers={"Origin": "http://example.com"})
    assert r.status_code == 200
    assert "access-control-allow-origin" in r.headers


def test_cors_does_not_combine_wildcard_with_credentials(client):
    # allow_origins=["*"] + allow_credentials=True is forbidden by the CORS
    # spec and would let any origin make credentialed requests.
    r = client.get("/api/health", headers={"Origin": "http://example.com"})
    assert r.headers.get("access-control-allow-credentials") != "true"


def test_unauthenticated_request_rejected_when_api_key_set(run_dir, monkeypatch):
    monkeypatch.setenv("TRAINSCOPE_API_KEY", "secret-key")
    client = TestClient(create_app(str(run_dir)))

    r = client.get("/api/global")
    assert r.status_code == 401


def test_authenticated_request_allowed_with_api_key(run_dir, monkeypatch):
    monkeypatch.setenv("TRAINSCOPE_API_KEY", "secret-key")
    client = TestClient(create_app(str(run_dir)))

    r = client.get("/api/global", headers={"Authorization": "Bearer secret-key"})
    assert r.status_code == 200


def test_health_stays_public_when_api_key_set(run_dir, monkeypatch):
    monkeypatch.setenv("TRAINSCOPE_API_KEY", "secret-key")
    client = TestClient(create_app(str(run_dir)))

    r = client.get("/api/health")
    assert r.status_code == 200


def test_websocket_rejected_without_api_key(run_dir, monkeypatch):
    monkeypatch.setenv("TRAINSCOPE_API_KEY", "secret-key")
    client = TestClient(create_app(str(run_dir)))

    with pytest.raises(Exception):
        with client.websocket_connect("/ws"):
            pass


def test_websocket_allowed_with_api_key(run_dir, monkeypatch):
    monkeypatch.setenv("TRAINSCOPE_API_KEY", "secret-key")
    client = TestClient(create_app(str(run_dir)))

    with client.websocket_connect("/ws", headers={"Authorization": "Bearer secret-key"}) as ws:
        assert ws.receive_json()["type"] == "meta"


def test_server_reads_stream_format_run(tmp_path):
    """A run written by the append-only DiskWriter (IPC stream format) must be
    served by the same API as a legacy run."""
    from trainscope.core.config import TrainScopeConfig
    from trainscope.io.writer import DiskWriter

    run = tmp_path / "stream_run"
    config = TrainScopeConfig(
        run_dir=str(tmp_path), run_name="stream_run", compaction_every_n_steps=6
    )
    writer = DiskWriter(run, config)
    for i in range(10):
        writer.append_global(
            {
                "step": i,
                "loss": float(i),
                "grad_norm_before_clip": 0.5,
                "grad_norm_after_clip": 0.5,
                "lr": 0.001,
                "optimizer_v_norm": 0.0,
                "step_time_ms": 1.0,
                "batch_index": i,
                "is_spike": False,
                "cpu_memory_mb": 0.0,
                "cuda_memory_mb": 0.0,
                "spike_score": 0.0,
            }
        )
    writer.append_layer("layer0", {"step": 0, "grad_l2_norm": 0.1})
    writer.close()

    client = TestClient(create_app(str(run)))
    global_rows = client.get("/api/global").json()
    assert len(global_rows) == 10
    assert [r["step"] for r in global_rows] == list(range(10))
    assert client.get("/api/layers").json() == ["layer0"]
    layer_rows = client.get("/api/layers/layer0").json()
    assert layer_rows[0]["step"] == 0
    assert layer_rows[0]["grad_l2_norm"] == 0.1
    assert layer_rows[0]["act_kurtosis"] is None


class TestMultiRun:
    def test_runs_lists_single_run_when_no_root(self, client):
        """Single-run mode exposes the current run as a one-element list."""
        runs = client.get("/api/runs").json()
        assert len(runs) == 1
        assert runs[0]["name"] == "run"
        assert runs[0]["model_name"] == "TestModel"
        assert runs[0]["n_global_rows"] == 5
        assert runs[0]["spike_count"] == 1
        assert runs[0]["last_loss"] == 4.0
        assert runs[0]["is_active"] is True

    def test_runs_lists_and_selects_across_root(self, run_dir, tmp_path):
        """--runs mode: /api/runs lists every run, select switches the active one."""
        root = tmp_path / "root"
        root.mkdir()

        import shutil

        shutil.copytree(run_dir, root / "run_a")
        # Second run with different loss and no spikes.
        run_b = root / "run_b"
        run_b.mkdir()
        run_b.joinpath("meta.json").write_text(
            json.dumps(
                {
                    "model_name": "TestModelB",
                    "model_config": {},
                    "trainscope_config": {"run_name": "run_b"},
                }
            )
        )
        rows = [
            {
                "step": i,
                "loss": 2.0 + 0.01 * i,
                "grad_norm_before_clip": 1.0,
                "grad_norm_after_clip": 1.0,
                "lr": 0.001,
                "optimizer_v_norm": 0.0,
                "step_time_ms": 1.0,
                "batch_index": i,
                "is_spike": False,
                "cpu_memory_mb": 0.0,
                "cuda_memory_mb": 0.0,
            }
            for i in range(10)
        ]
        _write_arrow(run_b / "global.arrow", GLOBAL_SCHEMA, rows)
        run_b.joinpath("manifest.json").write_text(
            json.dumps({"last_step": 9, "n_global_rows": 10})
        )

        client = TestClient(create_app(str(root / "run_a"), runs_root=str(root)))
        runs = client.get("/api/runs").json()
        names = {r["name"] for r in runs}
        assert names == {"run_a", "run_b"}

        by_name = {r["name"]: r for r in runs}
        assert by_name["run_a"]["is_active"] is True
        assert by_name["run_a"]["spike_count"] == 1
        assert by_name["run_b"]["spike_count"] == 0
        assert by_name["run_b"]["n_global_rows"] == 10

        # Default active run is the first discovered run (run_a).
        assert client.get("/api/global").json()[0]["step"] == 0
        assert client.get("/api/meta").json()["model_name"] == "TestModel"

        # Switch to run_b: meta/global now point at it.
        resp = client.post("/api/runs/select", json={"name": "run_b"})
        assert resp.status_code == 200
        assert resp.json()["name"] == "run_b"
        assert resp.json()["is_active"] is True

        meta = client.get("/api/meta").json()
        assert meta["model_name"] == "TestModelB"
        global_rows = client.get("/api/global").json()
        assert len(global_rows) == 10

        # Unknown run -> 404.
        assert client.post("/api/runs/select", json={"name": "nope"}).status_code == 404

    def test_select_mismatch_in_single_run_mode(self, client):
        resp = client.post("/api/runs/select", json={"name": "other"})
        assert resp.status_code == 400


class TestCompare:
    def _make_root(self, tmp_path):
        root = tmp_path / "cmp_root"
        root.mkdir()
        return root

    def _write_run(self, path, rows, meta=None):
        path.mkdir(parents=True, exist_ok=True)
        path.joinpath("meta.json").write_text(
            json.dumps(
                meta
                or {
                    "model_name": "M",
                    "model_config": {"d_model": 128},
                    "trainscope_config": {"run_name": path.name, "full_resolution_window": 500},
                }
            )
        )
        _write_arrow(path / "global.arrow", GLOBAL_SCHEMA, rows)
        path.joinpath("manifest.json").write_text(
            json.dumps({"last_step": rows[-1]["step"], "n_global_rows": len(rows)})
        )

    def _row(self, step, loss, is_spike=False):
        return {
            "step": step,
            "loss": loss,
            "grad_norm_before_clip": 1.0,
            "grad_norm_after_clip": 1.0,
            "lr": 0.001,
            "optimizer_v_norm": 0.0,
            "step_time_ms": 1.0,
            "batch_index": step,
            "is_spike": is_spike,
            "cpu_memory_mb": 0.0,
            "cuda_memory_mb": 0.0,
        }

    def test_compare_requires_multi_run_mode(self, client):
        r = client.get("/api/compare", params={"runs": "a,b"})
        assert r.status_code == 404

    def test_compare_finds_divergence_and_config_diff(self, tmp_path):
        root = self._make_root(tmp_path)
        # Run A: flat 1.0 loss the whole way. Run B: same until step 40, then
        # drifts upward — a durable separation after a shared warmup.
        rows_a = [self._row(i, 1.0) for i in range(80)]
        rows_b = [self._row(i, 1.0) for i in range(40)] + [
            self._row(i, 1.0 + 0.05 * (i - 39)) for i in range(40, 80)
        ]
        self._write_run(root / "run_a", rows_a)
        self._write_run(
            root / "run_b",
            rows_b,
            meta={
                "model_name": "M",
                "model_config": {"d_model": 256},  # differs from run_a's 128
                "trainscope_config": {
                    "run_name": "run_b",
                    "full_resolution_window": 1000,  # differs
                },
            },
        )

        client = TestClient(create_app(str(root / "run_a"), runs_root=str(root)))
        r = client.get("/api/compare", params={"runs": "run_a,run_b"})
        assert r.status_code == 200
        data = r.json()

        assert data["runs"] == ["run_a", "run_b"]
        assert data["divergence"]["step"] == 40
        assert data["divergence"]["min_run"] == 3

        fields = {d["field"]: d["values"] for d in data["config_diff"]}
        assert fields["config.full_resolution_window"] == {"run_a": 500, "run_b": 1000}
        assert fields["model.d_model"] == {"run_a": 128, "run_b": 256}
        # Identical fields are not reported.
        assert all("run_name" not in f for f in fields)

    def test_compare_common_cause_threshold(self, tmp_path):
        root = self._make_root(tmp_path)
        # Two runs with spikes share lr=0.001; one stable run has lr=0.0001.
        for name, lr, spike in [
            ("spike_a", 0.001, True),
            ("spike_b", 0.001, True),
            ("ok_c", 0.0001, False),
        ]:
            rows = [self._row(i, 1.0, is_spike=spike and i == 30) for i in range(60)]
            self._write_run(
                root / name,
                rows,
                meta={
                    "model_name": "M",
                    "model_config": {},
                    "trainscope_config": {
                        "run_name": name,
                        "full_resolution_window": 500,
                        "detector": {"name": "changepoint", "threshold": lr},
                    },
                },
            )
            if spike:
                spikes = root / name / "spikes"
                spikes.mkdir()
                _write_arrow(spikes / "spike_step_30.arrow", GLOBAL_SCHEMA, [rows[30]])
        client = TestClient(create_app(str(root / "spike_a"), runs_root=str(root)))
        data = client.get("/api/compare", params={"runs": "spike_a,spike_b,ok_c"}).json()
        causes = {c["field"]: c for c in data["common_cause"]}
        assert "config.detector.threshold" in causes
        cause = causes["config.detector.threshold"]
        assert cause["spiked_value"] == 0.001
        assert cause["stable_value"] == 0.0001

    def test_compare_missing_run_404(self, tmp_path):
        root = self._make_root(tmp_path)
        self._write_run(root / "run_a", [self._row(i, 1.0) for i in range(20)])
        client = TestClient(create_app(str(root / "run_a"), runs_root=str(root)))
        r = client.get("/api/compare", params={"runs": "run_a,ghost"})
        assert r.status_code == 404

    def test_compare_requires_two_runs(self, tmp_path):
        root = self._make_root(tmp_path)
        self._write_run(root / "run_a", [self._row(i, 1.0) for i in range(20)])
        client = TestClient(create_app(str(root / "run_a"), runs_root=str(root)))
        r = client.get("/api/compare", params={"runs": "run_a"})
        assert r.status_code == 400


class TestCompareConcentration:
    def test_compare_includes_concentration_series_and_common_cause(self, tmp_path):
        from trainscope.io.writer import MOE_SCHEMA

        root = tmp_path / "moecmp"
        root.mkdir()

        def write_run(name, spike, peak_share, det_name):
            path = root / name
            path.mkdir()
            path.joinpath("meta.json").write_text(
                json.dumps(
                    {
                        "model_name": "MoE",
                        "model_config": {},
                        "trainscope_config": {
                            "run_name": name,
                            "full_resolution_window": 500,
                            "detector": {"name": det_name, "threshold": 0.85},
                        },
                    }
                )
            )
            rows = []
            for i in range(40):
                loss = 1.0
                if spike and i == 30:
                    loss = 100.0
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
                        "is_spike": spike and i == 30,
                        "cpu_memory_mb": 0.0,
                        "cuda_memory_mb": 0.0,
                    }
                )
            _write_arrow(path / "global.arrow", GLOBAL_SCHEMA, rows)
            path.joinpath("manifest.json").write_text(
                json.dumps({"last_step": 39, "n_global_rows": 40})
            )
            if spike:
                spikes = path / "spikes"
                spikes.mkdir()
                _write_arrow(spikes / "spike_step_30.arrow", GLOBAL_SCHEMA, [rows[30]])

            # MoE routing: shares over 4 experts; concentrate (0.95) on spike
            # runs, diffuse (0.30) on stable runs.
            moe_rows = []
            for i in range(40):
                if spike and i >= 25:
                    shares = [0.95, 0.02, 0.02, 0.01]
                else:
                    shares = [0.30, 0.30, 0.25, 0.15]
                moe_rows.append({"step": i, "block": "blocks.0.router", "shares": shares})
            _write_arrow(path / "moe.arrow", MOE_SCHEMA, moe_rows)

        write_run("moe_spike", spike=True, peak_share=0.95, det_name="expert_utilization_drift")
        write_run("moe_ok", spike=False, peak_share=0.30, det_name="expert_utilization_drift")

        client = TestClient(create_app(str(root / "moe_spike"), runs_root=str(root)))
        data = client.get("/api/compare", params={"runs": "moe_spike,moe_ok"}).json()

        assert "concentration_series" in data
        assert data["concentration_series"]["moe_spike"][-1]["max_share"] == 0.95
        assert data["concentration_series"]["moe_ok"][-1]["max_share"] == 0.30

        causes = {c["field"]: c for c in data["common_cause"]}
        assert "max routing concentration" in causes
        assert causes["max routing concentration"]["spiked_value"] == 0.95
        assert causes["max routing concentration"]["stable_value"] == 0.30

    def test_compare_concentration_absent_for_non_moe_runs(self, tmp_path):
        root = tmp_path / "nonmoe"
        root.mkdir()
        for name in ("run_a", "run_b"):
            path = root / name
            path.mkdir()
            path.joinpath("meta.json").write_text(
                json.dumps(
                    {
                        "model_name": "M",
                        "model_config": {},
                        "trainscope_config": {
                            "run_name": name,
                            "detector": {"name": "changepoint"},
                        },
                    }
                )
            )
            _write_arrow(
                path / "global.arrow",
                GLOBAL_SCHEMA,
                [
                    {
                        "step": i,
                        "loss": 1.0,
                        "grad_norm_before_clip": 1.0,
                        "grad_norm_after_clip": 1.0,
                        "lr": 0.001,
                        "optimizer_v_norm": 0.0,
                        "step_time_ms": 1.0,
                        "batch_index": i,
                        "is_spike": False,
                        "cpu_memory_mb": 0.0,
                        "cuda_memory_mb": 0.0,
                    }
                    for i in range(20)
                ],
            )
        client = TestClient(create_app(str(root / "run_a"), runs_root=str(root)))
        data = client.get("/api/compare", params={"runs": "run_a,run_b"}).json()
        assert data["concentration_series"]["run_a"] == []
        assert not any(c["field"] == "max routing concentration" for c in data["common_cause"])


class TestCluster:
    def _write_layer_kurtosis(self, path, values):
        layers_dir = path / "layers"
        layers_dir.mkdir(exist_ok=True)
        rows = []
        for i, k in enumerate(values):
            rows.append(
                {
                    "step": i,
                    "grad_l2_norm": 1.0,
                    "weight_l2_norm": 1.0,
                    "act_mean": 0.0,
                    "act_std": 1.0,
                    "act_max_abs": 1.0,
                    "act_kurtosis": k,
                    "grad_nan_inf_ratio": 0.0,
                    "hist_counts": [],
                    "hist_edges": [],
                    "grad_max_abs": 1.0,
                    "grad_mean": 0.0,
                    "weight_mean": 0.0,
                    "weight_std": 1.0,
                    "weight_max_abs": 1.0,
                    "weight_min": -1.0,
                    "act_min": -1.0,
                    "act_max": 1.0,
                    "act_median": 0.0,
                }
            )
        _write_arrow(layers_dir / "layer0.arrow", LAYER_SCHEMA, rows)

    def _write_run(
        self,
        root,
        name,
        grad_spike=False,
        kurtosis_spike=False,
        loss_spike=False,
        lr=None,
        loss_scale=1.0,
    ):
        path = root / name
        path.mkdir()
        path.joinpath("meta.json").write_text(
            json.dumps(
                {
                    "model_name": "M",
                    "model_config": {"lr": lr} if lr is not None else {},
                    "trainscope_config": {
                        "run_name": name,
                        "full_resolution_window": 500,
                        "detector": {"name": "changepoint", "threshold": 6.0},
                    },
                }
            )
        )
        rows = []
        for i in range(80):
            loss = 1.0 * loss_scale
            grad = 1.0
            if loss_spike and i == 60:
                loss = 100.0
            if grad_spike and i >= 40:
                grad = 50.0
            rows.append(
                {
                    "step": i,
                    "loss": loss,
                    "grad_norm_before_clip": grad,
                    "grad_norm_after_clip": grad,
                    "lr": 0.001,
                    "optimizer_v_norm": 0.0,
                    "step_time_ms": 1.0,
                    "batch_index": i,
                    "is_spike": loss_spike and i == 60,
                    "cpu_memory_mb": 0.0,
                    "cuda_memory_mb": 0.0,
                }
            )
        _write_arrow(path / "global.arrow", GLOBAL_SCHEMA, rows)
        if loss_spike:
            spikes = path / "spikes"
            spikes.mkdir()
            _write_arrow(spikes / "spike_step_60.arrow", GLOBAL_SCHEMA, [rows[60]])

        if kurtosis_spike:
            self._write_layer_kurtosis(path, [0.2] * 40 + [3.0] * 40)
        else:
            self._write_layer_kurtosis(path, [0.2] * 80)
        return path

    def test_cluster_groups_by_signal_signature(self, tmp_path):
        root = tmp_path / "cl"
        root.mkdir()
        # Two gradient-led runs (grad spike, no loss spike), one loss-led.
        self._write_run(root, "grad_a", grad_spike=True)
        self._write_run(root, "grad_b", grad_spike=True)
        self._write_run(root, "loss_c", loss_spike=True)
        self._write_run(root, "calm_d")

        client = TestClient(create_app(str(root / "grad_a"), runs_root=str(root)))
        data = client.get("/api/cluster").json()

        clusters = {c["label"]: c for c in data["clusters"]}
        assert "gradient-led" in clusters
        assert set(clusters["gradient-led"]["runs"]) == {"grad_a", "grad_b"}
        assert "loss-led" in clusters
        assert clusters["loss-led"]["runs"] == ["loss_c"]
        assert "no-signal" in clusters or "calm_d" in data["unclustered"]

    def test_counterexample_finds_nearest_stable_run(self, tmp_path):
        """An exploding run gets its nearest stable run (by config distance):
        the loss series of both, the first fired signal's crossing step, and
        the config fields that differ between the pair."""
        root = tmp_path / "ce"
        root.mkdir()
        # grad_a is gradient-led at lr=5e-4. Two stable runs exist: one at
        # lr=5e-4 (identical config, closest) and one at lr=1e-4 (farther).
        self._write_run(root, "grad_a", grad_spike=True, lr=5e-4)
        self._write_run(root, "stable_same", lr=5e-4)
        self._write_run(root, "stable_far", lr=1e-4)

        client = TestClient(create_app(str(root / "grad_a"), runs_root=str(root)))
        data = client.get("/api/counterexample?run=grad_a").json()

        assert data["counterexample_run"] == "stable_same"
        assert data["config_distance"] == 0.0
        assert data["first_signal"] == "grad_norm"
        assert data["crossing_step"] == 40
        assert set(data["loss_series"]) == {"grad_a", "stable_same"}
        assert len(data["loss_series"]["grad_a"]) == 80
        # Identical configs -> no differing fields.
        assert data["config_diff"] == []

    def test_counterexample_reports_differing_config(self, tmp_path):
        """When the nearest stable run differs in a scalar field, that field
        is the candidate cause and is reported."""
        root = tmp_path / "ce2"
        root.mkdir()
        self._write_run(root, "grad_a", grad_spike=True, lr=5e-4)
        self._write_run(root, "stable_low", lr=1e-4)

        client = TestClient(create_app(str(root / "grad_a"), runs_root=str(root)))
        data = client.get("/api/counterexample?run=grad_a").json()

        assert data["counterexample_run"] == "stable_low"
        diff = {d["field"]: d for d in data["config_diff"]}
        assert "model.lr" in diff
        assert diff["model.lr"]["query_value"] == 5e-4
        assert diff["model.lr"]["stable_value"] == 1e-4
        # The two runs differ in lr, so config distance is non-zero.
        assert data["config_distance"] > 0.0

    def test_counterexample_requires_spiking_run(self, tmp_path):
        """A run with no fired signal and no spikes has no counterexample."""
        root = tmp_path / "ce3"
        root.mkdir()
        self._write_run(root, "calm_a")
        self._write_run(root, "calm_b")

        client = TestClient(create_app(str(root / "calm_a"), runs_root=str(root)))
        assert client.get("/api/counterexample?run=calm_a").status_code == 404

    def test_counterexample_requires_multi_run_mode(self, client):
        assert client.get("/api/counterexample?run=run").status_code == 404

    def test_cluster_loss_band(self, tmp_path):
        """The cluster's loss band is a per-step median + IQR of each member's
        loss normalized by its own first-40 baseline mean. Two runs on very
        different absolute loss scales but the same shape must produce a
        narrow (near-degenerate) band: normalization aligns them."""
        root = tmp_path / "clb"
        root.mkdir()
        # big_b's raw loss is 10x big_a's, but both are constant curves, so
        # normalized to their own baseline they coincide at every step.
        self._write_run(root, "big_a", grad_spike=True, loss_scale=1.0)
        self._write_run(root, "big_b", grad_spike=True, loss_scale=10.0)

        client = TestClient(create_app(str(root / "big_a"), runs_root=str(root)))
        data = client.get("/api/cluster").json()
        grad = [c for c in data["clusters"] if c["label"] == "gradient-led"][0]

        band = grad["loss_band"]
        assert len(band["steps"]) >= 2
        assert len(band["median"]) == len(band["steps"])
        assert len(band["lower"]) == len(band["steps"])
        assert len(band["upper"]) == len(band["steps"])
        # Identical normalized curves -> degenerate band everywhere.
        assert band["lower"] == band["median"] == band["upper"]
        # A constant loss normalized by its own baseline mean sits at ~1.0.
        assert all(abs(v - 1.0) < 1e-6 for v in band["median"])

    def test_cluster_requires_multi_run_mode(self, client):
        assert client.get("/api/cluster").status_code == 404

    def test_cluster_discriminant_traits(self, tmp_path):
        """A config field shared by every run in a cluster but absent from the
        stable runs is reported as a discriminant trait: 'why did THIS failure
        mode blow up and not the others' at a glance."""
        root = tmp_path / "cld"
        root.mkdir()
        # Two gradient-led runs at a high LR, one loss-led at a low LR, one
        # stable run at a low LR. The gradient cluster shares lr=5e-4, which
        # no stable run has; the loss-led run's lr matches stable runs, so it
        # must NOT report that trait.
        self._write_run(root, "grad_a", grad_spike=True, lr=5e-4)
        self._write_run(root, "grad_b", grad_spike=True, lr=5e-4)
        self._write_run(root, "loss_c", loss_spike=True, lr=1e-4)
        self._write_run(root, "calm_d", lr=1e-4)

        client = TestClient(create_app(str(root / "grad_a"), runs_root=str(root)))
        data = client.get("/api/cluster").json()
        clusters = {c["label"]: c for c in data["clusters"]}

        grad = clusters["gradient-led"]
        assert grad["discriminant_traits"] == [{"field": "model.lr", "value": 5e-4}]

        loss = clusters["loss-led"]
        # loss_c shares lr=1e-4 with the stable run -> not discriminant.
        assert loss["discriminant_traits"] == []

    def test_cluster_detects_kurtosis_led(self, tmp_path):
        root = tmp_path / "clk"
        root.mkdir()
        self._write_run(root, "kurt_a", kurtosis_spike=True)
        self._write_run(root, "calm_b")

        client = TestClient(create_app(str(root / "kurt_a"), runs_root=str(root)))
        data = client.get("/api/cluster").json()
        labels = {c["label"] for c in data["clusters"]}
        assert "activation-led" in labels


class TestClusterChronology:
    """Multi-signal runs: the 'first' label must reflect real crossing order."""

    def test_first_signal_is_chronological_not_code_order(self, tmp_path):
        """Kurtosis crosses at step 30, grad norm at step 50 — kurtosis must
        be labeled first even though the code checks grad_norm before
        kurtosis."""
        root = tmp_path / "chrono"
        root.mkdir()

        path = root / "multi"
        path.mkdir()
        path.joinpath("meta.json").write_text(
            json.dumps(
                {
                    "model_name": "M",
                    "model_config": {},
                    "trainscope_config": {
                        "run_name": "multi",
                        "full_resolution_window": 500,
                        "detector": {"name": "changepoint", "threshold": 6.0},
                    },
                }
            )
        )
        rows = []
        for i in range(80):
            grad = 1.0
            if i >= 50:
                grad = 50.0  # grad norm crosses later (step ~50)
            rows.append(
                {
                    "step": i,
                    "loss": 1.0,
                    "grad_norm_before_clip": grad,
                    "grad_norm_after_clip": grad,
                    "lr": 0.001,
                    "optimizer_v_norm": 0.0,
                    "step_time_ms": 1.0,
                    "batch_index": i,
                    "is_spike": False,
                    "cpu_memory_mb": 0.0,
                    "cuda_memory_mb": 0.0,
                }
            )
        _write_arrow(path / "global.arrow", GLOBAL_SCHEMA, rows)

        # Kurtosis crosses EARLIER (step ~30).
        layers_dir = path / "layers"
        layers_dir.mkdir()
        layer_rows = []
        for i in range(80):
            k = 3.0 if i >= 30 else 0.2
            layer_rows.append(
                {
                    "step": i,
                    "grad_l2_norm": 1.0,
                    "weight_l2_norm": 1.0,
                    "act_mean": 0.0,
                    "act_std": 1.0,
                    "act_max_abs": 1.0,
                    "act_kurtosis": k,
                    "grad_nan_inf_ratio": 0.0,
                    "hist_counts": [],
                    "hist_edges": [],
                    "grad_max_abs": 1.0,
                    "grad_mean": 0.0,
                    "weight_mean": 0.0,
                    "weight_std": 1.0,
                    "weight_max_abs": 1.0,
                    "weight_min": -1.0,
                    "act_min": -1.0,
                    "act_max": 1.0,
                    "act_median": 0.0,
                }
            )
        _write_arrow(layers_dir / "layer0.arrow", LAYER_SCHEMA, layer_rows)

        client = TestClient(create_app(str(path), runs_root=str(root)))
        data = client.get("/api/cluster").json()

        assert len(data["clusters"]) == 1
        cluster = data["clusters"][0]
        # Both signals fired; the earlier one (kurtosis) anchors the label.
        assert set(cluster["fired_signals"]) == {"kurtosis", "grad_norm"}
        assert cluster["first_signal"] == "kurtosis"
        assert cluster["label"] == "activation-led"


class TestClusterLead:
    def test_typical_lead_uses_explosion_minus_first_crossing(self, tmp_path):
        """Loss spike at step 60, kurtosis crossing at step 45 -> lead 15."""
        root = tmp_path / "lead"
        root.mkdir()
        path = root / "run_a"
        path.mkdir()
        path.joinpath("meta.json").write_text(
            json.dumps(
                {
                    "model_name": "M",
                    "model_config": {},
                    "trainscope_config": {
                        "run_name": "run_a",
                        "full_resolution_window": 500,
                        "detector": {"name": "changepoint", "threshold": 6.0},
                    },
                }
            )
        )
        rows = []
        for i in range(80):
            loss = 100.0 if i == 60 else 1.0
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
                    "is_spike": i == 60,
                    "cpu_memory_mb": 0.0,
                    "cuda_memory_mb": 0.0,
                }
            )
        _write_arrow(path / "global.arrow", GLOBAL_SCHEMA, rows)
        spikes = path / "spikes"
        spikes.mkdir()
        _write_arrow(spikes / "spike_step_60.arrow", GLOBAL_SCHEMA, [rows[60]])

        # Kurtosis crosses at step 45 (earlier than the loss spike; the
        # crossing scan only starts after the 40-sample baseline).
        layers_dir = path / "layers"
        layers_dir.mkdir()
        layer_rows = []
        for i in range(80):
            layer_rows.append(
                {
                    "step": i,
                    "grad_l2_norm": 1.0,
                    "weight_l2_norm": 1.0,
                    "act_mean": 0.0,
                    "act_std": 1.0,
                    "act_max_abs": 1.0,
                    "act_kurtosis": 3.0 if i >= 45 else 0.2,
                    "grad_nan_inf_ratio": 0.0,
                    "hist_counts": [],
                    "hist_edges": [],
                    "grad_max_abs": 1.0,
                    "grad_mean": 0.0,
                    "weight_mean": 0.0,
                    "weight_std": 1.0,
                    "weight_max_abs": 1.0,
                    "weight_min": -1.0,
                    "act_min": -1.0,
                    "act_max": 1.0,
                    "act_median": 0.0,
                }
            )
        _write_arrow(layers_dir / "layer0.arrow", LAYER_SCHEMA, layer_rows)

        client = TestClient(create_app(str(path), runs_root=str(root)))
        data = client.get("/api/cluster").json()
        assert len(data["clusters"]) == 1
        cluster = data["clusters"][0]
        assert cluster["first_signal"] == "kurtosis"
        # explosion (60) - first crossing (45) = 15.
        assert cluster["typical_lead_steps"] == 15.0
