import json
import pickle

import pyarrow.ipc as ipc
import torch

from trainscope.core.config import TrainScopeConfig
from trainscope.io.writer import DiskWriter


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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        assert "trainscope_config" in meta
        assert "start_time" in meta

    def test_write_meta_includes_detector_info(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        writer.write_meta(
            "TestModel",
            {"layers": 12},
            detector_info={"name": "changepoint", "min_observations": 30},
        )
        writer.close()

        with open(run_path / "meta.json") as f:
            meta = json.load(f)
        assert meta["detector"] == {"name": "changepoint", "min_observations": 30}

    def test_layer_null_activation_metrics_round_trip(self, tmp_path):
        """Unmeasured activation metrics must persist as null, not as 0.0,
        so the UI can distinguish 'not measured' from 'measured zero'."""
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        measured = make_layer_snap(0)
        measured["act_kurtosis"] = 2.5
        unmeasured = make_layer_snap(1)
        unmeasured["act_mean"] = None
        unmeasured["act_std"] = None
        unmeasured["act_max_abs"] = None
        unmeasured["act_kurtosis"] = None
        unmeasured["act_min"] = None
        unmeasured["act_max"] = None
        unmeasured["act_median"] = None

        writer.append_layer("layer0", measured)
        writer.append_layer("layer0", unmeasured)
        writer.flush()
        writer.close()

        reader = ipc.open_file(str(run_path / "layers" / "layer0.arrow"))
        table = reader.read_all()
        assert table.column("act_kurtosis").to_pylist() == [2.5, None]
        assert table.column("act_mean").to_pylist() == [0.0, None]
        assert table.column("act_median").to_pylist() == [None, None]

    def test_append_global_flush_creates_arrow(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        writer.append_layer("transformer/h/0/attn", make_layer_snap(0))
        writer.flush()
        writer.close()

        safe_file = run_path / "layers" / "transformer%2Fh%2F0%2Fattn.arrow"
        assert safe_file.exists()

    def test_save_rng_state(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        writer.close()

        assert (run_path / "layers").is_dir()
        assert (run_path / "spikes").is_dir()
        assert (run_path / "rng_states").is_dir()

    def test_write_spike_window_writes_layer_data(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
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
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        for name in ["layer0.weight", "layer1.weight", "layer2.weight"]:
            for i in range(2):
                writer.append_layer(name, make_layer_snap(i))
        writer.flush()
        writer.close()

        for name in ["layer0.weight", "layer1.weight", "layer2.weight"]:
            assert (run_path / "layers" / f"{name}.arrow").exists()

    def test_context_manager_closes_writer(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        with DiskWriter(run_path, config) as writer:
            writer.append_global(make_global_snap(0))
        assert writer._closed is True
        assert (run_path / "global.arrow").exists()

    def test_layer_name_encoding_reversible(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)

        # Pathological name with slashes, dots, and double underscores.
        name = "transformer/h__0/attn.c_proj"
        writer.append_layer(name, make_layer_snap(0))
        writer.flush()
        writer.close()

        encoded = DiskWriter._encode_layer_name(name)
        assert DiskWriter._decode_layer_name(encoded + ".arrow") == name
        assert (run_path / "layers" / f"{encoded}.arrow").exists()

    def test_resume_appends_global_rows(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"

        writer = DiskWriter(run_path, config)
        for i in range(5):
            writer.append_global(make_global_snap(i))
        writer.close()

        config_resume = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run", resume=True)
        writer2 = DiskWriter(run_path, config_resume)
        for i in range(5, 10):
            writer2.append_global(make_global_snap(i))
        writer2.flush()
        writer2.close()

        reader = ipc.open_file(str(run_path / "global.arrow"))
        table = reader.read_all()
        assert table.num_rows == 10
        assert list(table.column("step").to_pylist()) == list(range(10))

    def test_resume_preserves_rows_across_multiple_flushes(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"

        writer = DiskWriter(run_path, config)
        for i in range(3):
            writer.append_global(make_global_snap(i))
        writer.close()

        config_resume = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run", resume=True)
        writer2 = DiskWriter(run_path, config_resume)
        for i in range(3, 6):
            writer2.append_global(make_global_snap(i))
        writer2.flush()
        for i in range(6, 9):
            writer2.append_global(make_global_snap(i))
        writer2.flush()
        writer2.close()

        reader = ipc.open_file(str(run_path / "global.arrow"))
        table = reader.read_all()
        assert table.num_rows == 9
        assert list(table.column("step").to_pylist()) == list(range(9))

    def test_resume_appends_layer_rows(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"

        writer = DiskWriter(run_path, config)
        writer.append_layer("layer0", make_layer_snap(0))
        writer.append_layer("layer0", make_layer_snap(1))
        writer.close()

        config_resume = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run", resume=True)
        writer2 = DiskWriter(run_path, config_resume)
        writer2.append_layer("layer0", make_layer_snap(2))
        writer2.flush()
        writer2.close()

        reader = ipc.open_file(str(run_path / "layers" / "layer0.arrow"))
        table = reader.read_all()
        assert table.num_rows == 3

    def test_manifest_written_on_close(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        for i in range(3):
            writer.append_global(make_global_snap(i))
        writer.append_layer("fc.weight", make_layer_snap(0))
        writer.close()

        manifest_file = run_path / "manifest.json"
        assert manifest_file.exists()
        with open(manifest_file) as f:
            manifest = json.load(f)
        assert manifest["n_global_rows"] == 3
        assert manifest["last_step"] == 2
        assert "layer_files" in manifest
        assert "fc.weight" in manifest["layer_files"]

    def test_save_checkpoint(self, tmp_path):
        config = TrainScopeConfig(run_dir=str(tmp_path), run_name="test_run")
        run_path = tmp_path / "test_run"
        writer = DiskWriter(run_path, config)
        state = {"param": torch.tensor([1.0, 2.0])}
        writer.save_checkpoint(7, state, optimizer_state={"lr": 0.1})
        writer.close()

        ckpt_file = run_path / "checkpoints" / "7.pt"
        assert ckpt_file.exists()
        loaded = torch.load(str(ckpt_file), weights_only=False)
        assert loaded["step"] == 7
        assert "model_state_dict" in loaded
        assert "optimizer_state_dict" in loaded
