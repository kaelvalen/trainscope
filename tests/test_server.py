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
