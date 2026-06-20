import pytest

from trainscope.core.config import TrainScopeConfig


class TestTrainScopeConfig:
    def test_run_name_auto_generated(self):
        cfg = TrainScopeConfig()
        assert cfg.run_name is not None
        assert cfg.run_name.startswith("run_")

    def test_full_resolution_window_validation(self):
        with pytest.raises(ValueError, match="full_resolution_window"):
            TrainScopeConfig(full_resolution_window=0)

    def test_decimation_factor_validation(self):
        with pytest.raises(ValueError, match="decimation_factor"):
            TrainScopeConfig(decimation_factor=0)

    def test_spike_window_before_must_fit(self):
        with pytest.raises(ValueError, match="spike_window_before"):
            TrainScopeConfig(full_resolution_window=10, spike_window_before=11)

    def test_spike_window_after_non_negative(self):
        with pytest.raises(ValueError, match="spike_window_after"):
            TrainScopeConfig(spike_window_after=-1)

    @pytest.mark.parametrize(
        "field",
        [
            "histogram_every_n_steps",
            "activation_metrics_every_n_steps",
            "trace_every_n_steps",
        ],
    )
    def test_every_n_steps_positive(self, field):
        with pytest.raises(ValueError, match=field):
            TrainScopeConfig(**{field: 0})  # type: ignore[arg-type]

    def test_n_histogram_bins_validation(self):
        with pytest.raises(ValueError, match="n_histogram_bins"):
            TrainScopeConfig(n_histogram_bins=1)

    def test_rng_every_n_steps_non_negative(self):
        with pytest.raises(ValueError, match="rng_every_n_steps"):
            TrainScopeConfig(rng_every_n_steps=-1)

    def test_to_dict_round_trip(self):
        cfg = TrainScopeConfig(
            run_dir="./runs",
            run_name="test",
            device="cpu",
            track_memory=False,
            checkpoint_on_spike=True,
            rng_every_n_steps=10,
            resume=True,
        )
        d = cfg.to_dict()
        assert d["run_dir"] == "./runs"
        assert d["run_name"] == "test"
        assert d["device"] == "cpu"
        assert d["track_memory"] is False
        assert d["checkpoint_on_spike"] is True
        assert d["rng_every_n_steps"] == 10
        assert d["resume"] is True

    def test_device_none_serializes_to_none(self):
        cfg = TrainScopeConfig(device=None)
        assert cfg.to_dict()["device"] is None
