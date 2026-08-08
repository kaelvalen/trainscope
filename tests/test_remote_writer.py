"""Tests for the remote (fsspec-backed) writer."""

import pytest

from trainscope.core.config import TrainScopeConfig
from trainscope.io.remote_writer import RemoteWriter
from trainscope.io.writer import read_arrow_rows_bytes


@pytest.fixture(autouse=True)
def _clean_memory_fs():
    import fsspec

    store = fsspec.filesystem("memory").store
    store.clear()
    yield
    store.clear()


def make_global_snap(step: int = 0):
    return {
        "step": step,
        "loss": 1.23,
        "grad_norm_before_clip": 0.5,
        "grad_norm_after_clip": 0.5,
        "lr": 1e-3,
        "optimizer_v_norm": 0.1,
        "step_time_ms": 10.0,
        "batch_index": step,
        "is_spike": False,
        "cpu_memory_mb": 0.0,
        "cuda_memory_mb": 0.0,
        "spike_score": 0.0,
    }


def make_layer_snap(step: int = 0):
    return {
        "step": step,
        "grad_l2_norm": 0.3,
        "weight_l2_norm": 1.2,
        "act_mean": None,
        "act_std": None,
        "act_max_abs": None,
        "act_kurtosis": None,
        "grad_nan_inf_ratio": 0.0,
        "hist_counts": [float(i) for i in range(16)],
        "hist_edges": [float(i) * 0.1 for i in range(17)],
        "grad_max_abs": 0.3,
        "grad_mean": 0.1,
        "weight_mean": 0.0,
        "weight_std": 1.0,
        "weight_max_abs": 2.0,
        "weight_min": -1.0,
        "act_min": None,
        "act_max": None,
        "act_median": None,
    }


def read_remote(uri: str, path: str) -> list[dict]:
    import fsspec

    fs, _ = fsspec.core.url_to_fs(uri)
    with fs.open(path, "rb") as f:
        return read_arrow_rows_bytes(f.read())


class TestRemoteWriter:
    def test_write_and_read_roundtrip(self):
        config = TrainScopeConfig(run_dir="unused", run_name="r", storage_uri="memory://run")
        writer = RemoteWriter("memory://run", config)
        for i in range(5):
            writer.append_global(make_global_snap(i))
        writer.close()

        rows = read_remote("memory://run", "run/global.arrow")
        assert [r["step"] for r in rows] == list(range(5))

    def test_compaction_cadence_writes_periodically(self):
        config = TrainScopeConfig(
            run_dir="unused",
            run_name="r",
            storage_uri="memory://run",
            compaction_every_n_steps=6,
        )
        writer = RemoteWriter("memory://run", config)

        # Below the cadence: the object is not written yet.
        for i in range(5):
            writer.append_global(make_global_snap(i))
        fs = writer._fs
        assert not fs.exists(writer._global_path())

        # The 6th row crosses the cadence: full object written, all rows present.
        writer.append_global(make_global_snap(5))
        assert fs.exists(writer._global_path())
        rows = read_remote("memory://run", "run/global.arrow")
        assert [r["step"] for r in rows] == list(range(6))

        writer.append_global(make_global_snap(6))
        rows = read_remote("memory://run", "run/global.arrow")
        assert [r["step"] for r in rows] == list(range(6))

        writer.append_global(make_global_snap(7))
        writer.flush()
        rows = read_remote("memory://run", "run/global.arrow")
        assert [r["step"] for r in rows] == list(range(8))
        writer.close()

    def test_resume_stream_format(self):
        config = TrainScopeConfig(run_dir="unused", run_name="r", storage_uri="memory://run")
        writer = RemoteWriter("memory://run", config)
        for i in range(4):
            writer.append_global(make_global_snap(i))
        writer.append_layer("layer0", make_layer_snap(0))
        writer.close()

        resume = TrainScopeConfig(
            run_dir="unused", run_name="r", storage_uri="memory://run", resume=True
        )
        writer2 = RemoteWriter("memory://run", resume)
        assert [r["step"] for r in writer2._global_rows] == list(range(4))
        for i in range(4, 7):
            writer2.append_global(make_global_snap(i))
        writer2.flush()
        writer2.close()

        rows = read_remote("memory://run", "run/global.arrow")
        assert [r["step"] for r in rows] == list(range(7))

    def test_resume_legacy_file_format(self):
        """Objects written by trainscope <=0.7.0 use the legacy IPC file
        format; resuming must still load them."""
        import io

        import fsspec
        import pyarrow as pa
        import pyarrow.ipc as ipc

        fs, path = fsspec.core.url_to_fs("memory://run")
        fs.makedirs(path, exist_ok=True)

        table = pa.Table.from_pylist(
            [make_global_snap(0), make_global_snap(1)],
            schema=pa.schema(
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
                    pa.field("cpu_memory_mb", pa.float64()),
                    pa.field("cuda_memory_mb", pa.float64()),
                    pa.field("spike_score", pa.float64()),
                ]
            ),
        )
        bio = io.BytesIO()
        with ipc.new_file(bio, table.schema) as writer:
            writer.write_table(table)
        with fs.open(f"{path}/global.arrow", "wb") as f:
            f.write(bio.getvalue())

        resume = TrainScopeConfig(
            run_dir="unused", run_name="r", storage_uri="memory://run", resume=True
        )
        writer = RemoteWriter("memory://run", resume)
        assert [r["step"] for r in writer._global_rows] == [0, 1]
        for i in range(2, 5):
            writer.append_global(make_global_snap(i))
        writer.flush()
        writer.close()

        rows = read_remote("memory://run", "run/global.arrow")
        assert [r["step"] for r in rows] == list(range(5))

    def test_layer_and_plugin_metrics(self):
        config = TrainScopeConfig(
            run_dir="unused",
            run_name="r",
            storage_uri="memory://run",
            compaction_every_n_steps=5,
        )
        writer = RemoteWriter("memory://run", config)
        for i in range(8):
            writer.append_layer("layer0", make_layer_snap(i))
        writer.append_plugin_metrics(0, "p", {"m": 1.0})
        writer.close()

        rows = read_remote("memory://run", "run/layers/layer0.arrow")
        assert [r["step"] for r in rows] == list(range(8))

        prows = read_remote("memory://run", "run/plugin_metrics.arrow")
        assert [r["plugin"] for r in prows] == ["p"]

    def test_spike_window_written_once(self):
        config = TrainScopeConfig(run_dir="unused", run_name="r", storage_uri="memory://run")
        writer = RemoteWriter("memory://run", config)
        writer.write_spike_window(
            100,
            [{"global": make_global_snap(100)}],
            {"layer0": [make_layer_snap(100)]},
        )
        writer.close()

        rows = read_remote("memory://run", "run/spikes/spike_step_100.arrow")
        assert rows[0]["step"] == 100
        lrows = read_remote("memory://run", "run/spikes/spike_step_100_layers/layer0.arrow")
        assert lrows[0]["step"] == 100

    def test_write_meta_and_manifest(self):
        config = TrainScopeConfig(run_dir="unused", run_name="r", storage_uri="memory://run")
        writer = RemoteWriter("memory://run", config)
        writer.write_meta("TestModel", {"layers": 1}, detector_info={"name": "x"})
        for i in range(3):
            writer.append_global(make_global_snap(i))
        writer.close()

        import json

        import fsspec

        fs, _ = fsspec.core.url_to_fs("memory://run")
        with fs.open("run/meta.json", "r") as f:
            meta = json.load(f)
        assert meta["model_name"] == "TestModel"
        assert meta["detector"] == {"name": "x"}
        with fs.open("run/manifest.json", "r") as f:
            manifest = json.load(f)
        assert manifest["n_global_rows"] == 3


def test_remote_writer_requires_fsspec():
    config = TrainScopeConfig(run_dir="unused", run_name="r", storage_uri="s3://bucket/run")
    with pytest.raises(ImportError):
        RemoteWriter("s3://bucket/run", config)
