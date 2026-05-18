import json
import pickle

import pyarrow.ipc as ipc

from losswatch.core.config import LossWatchConfig
from losswatch.io.writer import DiskWriter


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
    }


def make_layer_snap(step: int = 0):
    return {
        "step": step,
        "grad_l2_norm": 0.3,
        "weight_l2_norm": 1.2,
        "act_mean": 0.0,
        "act_std": 1.0,
        "act_max_abs": 3.5,
        "act_kurtosis": 0.1,
        "grad_nan_inf_ratio": 0.0,
        "hist_counts": [float(i) for i in range(16)],
        "hist_edges": [float(i) * 0.1 for i in range(17)],
    }


class TestDiskWriter:
    def test_write_meta_creates_file(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        writer.write_meta("TestModel", {"layers": 12, "hidden": 768})
        writer.close()

        meta_file = run_path / "meta.json"
        assert meta_file.exists()
        with open(meta_file) as f:
            meta = json.load(f)
        assert meta["model_name"] == "TestModel"
        assert meta["model_config"]["layers"] == 12
        assert "losswatch_config" in meta
        assert "start_time" in meta

    def test_append_global_flush_creates_arrow(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        for i in range(5):
            writer.append_global(make_global_snap(i))
        writer.flush()
        writer.close()

        arrow_file = run_path / "global.arrow"
        assert arrow_file.exists()

        reader = ipc.open_file(str(arrow_file))
        table = reader.read_all()
        assert table.num_rows == 5
        assert "step" in table.schema.names
        assert "loss" in table.schema.names
        assert "is_spike" in table.schema.names

    def test_global_arrow_values_correct(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        snap = make_global_snap(42)
        snap["loss"] = 2.718
        snap["is_spike"] = True
        writer.append_global(snap)
        writer.flush()
        writer.close()

        reader = ipc.open_file(str(run_path / "global.arrow"))
        table = reader.read_all()
        d = table.to_pydict()
        assert d["step"][0] == 42
        assert abs(d["loss"][0] - 2.718) < 1e-6
        assert d["is_spike"][0] is True

    def test_append_layer_flush_creates_arrow(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        for i in range(3):
            writer.append_layer("transformer.layer0.weight", make_layer_snap(i))
        writer.flush()
        writer.close()

        layer_file = run_path / "layers" / "transformer.layer0.weight.arrow"
        assert layer_file.exists()

        reader = ipc.open_file(str(layer_file))
        table = reader.read_all()
        assert table.num_rows == 3
        assert "grad_l2_norm" in table.schema.names
        assert "hist_counts" in table.schema.names

    def test_layer_with_slash_in_name(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        writer.append_layer("transformer/h/0/attn", make_layer_snap(0))
        writer.flush()
        writer.close()

        safe_file = run_path / "layers" / "transformer__h__0__attn.arrow"
        assert safe_file.exists()

    def test_save_rng_state(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        writer.save_rng_state(99)
        writer.close()

        rng_file = run_path / "rng_states" / "step_99.pkl"
        assert rng_file.exists()
        with open(rng_file, "rb") as f:
            state = pickle.load(f)
        assert "torch_rng" in state
        assert "numpy_rng" in state

    def test_auto_flush_at_interval(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        for i in range(100):
            writer.append_global(make_global_snap(i))

        arrow_file = run_path / "global.arrow"
        assert arrow_file.exists()

        writer.close()

        reader = ipc.open_file(str(arrow_file))
        table = reader.read_all()
        assert table.num_rows == 100

    def test_directory_structure_created(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        writer.close()

        assert (run_path / "layers").is_dir()
        assert (run_path / "spikes").is_dir()
        assert (run_path / "rng_states").is_dir()

    def test_write_spike_window_writes_layer_data(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        global_snap = make_global_snap(100)
        layer_snap = make_layer_snap(100)

        window = [{"global": global_snap, "layers": {"fc.weight": layer_snap}, "step_number": 100}]
        layer_windows = {"fc.weight": [layer_snap]}

        writer.write_spike_window(100, window, layer_windows)
        writer.close()

        spike_global = run_path / "spikes" / "spike_step_100.arrow"
        assert spike_global.exists()

        layers_dir = run_path / "spikes" / "spike_step_100_layers"
        assert layers_dir.exists()
        layer_file = layers_dir / "fc.weight.arrow"
        assert layer_file.exists()

        reader = ipc.open_file(str(layer_file))
        table = reader.read_all()
        assert table.num_rows == 1
        assert "act_kurtosis" in table.schema.names
        assert "hist_counts" in table.schema.names

    def test_multiple_layers(self, tmp_path):
        config = LossWatchConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        for name in ["layer0.weight", "layer1.weight", "layer2.weight"]:
            for i in range(2):
                writer.append_layer(name, make_layer_snap(i))
        writer.flush()
        writer.close()

        for name in ["layer0.weight", "layer1.weight", "layer2.weight"]:
            assert (run_path / "layers" / f"{name}.arrow").exists()
