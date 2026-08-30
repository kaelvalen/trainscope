import pytest

from trainscope.core.config import TrainScopeConfig, load_config


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
            "compaction_every_n_steps",
        ],
    )
    def test_every_n_steps_positive(self, field):
        with pytest.raises(ValueError, match=field):
            TrainScopeConfig(**{field: 0})  # type: ignore[arg-type]

    def test_compaction_every_n_steps_default(self):
        assert TrainScopeConfig().compaction_every_n_steps == 1000
        assert TrainScopeConfig.from_env(prefix="X_").compaction_every_n_steps == 1000

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
        assert d["compaction_every_n_steps"] == 1000
        assert d["resume"] is True

    def test_device_none_serializes_to_none(self):
        cfg = TrainScopeConfig(device=None)
        assert cfg.to_dict()["device"] is None

    def test_spike_threshold_removed_from_top_level(self):
        """1.0 removed spike_threshold; detector thresholds live in detector dict."""
        cfg = TrainScopeConfig()
        assert "spike_threshold" not in cfg.to_dict()

        with pytest.raises(TypeError, match="spike_threshold"):
            TrainScopeConfig(spike_threshold=3.5)  # type: ignore[call-arg]

    def test_load_config_rejects_spike_threshold_with_migration_hint(self):
        from trainscope.core.config import load_config

        with pytest.raises(ValueError, match="detector="):
            load_config({"spike_threshold": 3.5})

    def test_detector_threshold_in_dict(self):
        cfg = TrainScopeConfig(detector={"name": "z_score", "threshold": 3.5})
        assert cfg.detector == {"name": "z_score", "threshold": 3.5}

    def test_z_score_detector_receives_threshold(self):
        from trainscope.core.detectors import make_detector
        from trainscope.core.detectors.z_score import ZScoreDetector

        cfg = TrainScopeConfig(detector={"name": "z_score", "threshold": 4.2})
        det = make_detector(cfg)
        assert isinstance(det, ZScoreDetector)
        assert det.threshold == 4.2


class TestConfigRoundTrip:
    """Phase 3: every config that can be written to a dict, YAML, or env must
    load back into an equivalent config. to_dict() output must be directly
    consumable by load_config(), and from_env()/from_yaml() must agree with
    their dict equivalents field for field."""

    def _equivalent(self, a: TrainScopeConfig, b: TrainScopeConfig) -> bool:
        da, db = a.to_dict(), b.to_dict()
        # run_name is generated at construction when unset; compare explicitly
        # so two constructions of an unset name still match.
        assert da.pop("run_name") == db.pop("run_name")
        return da == db

    def test_to_dict_feeds_load_config(self):
        """to_dict() is the canonical serialization; load_config() must accept
        it directly and reproduce the same config."""
        cfg = TrainScopeConfig(
            run_dir="./runs",
            run_name="roundtrip",
            detector={"name": "z_score", "threshold": 3.5},
            track_memory=False,
            rng_every_n_steps=7,
            checkpoint_on_spike="{step}_ckpt.pt",
            storage_uri="s3://bucket/runs",
            integrations={"wandb": {"alerts": True}},
            alerts=[{"type": "slack", "url": "https://example.com"}],
        )
        reloaded = load_config(cfg.to_dict())
        assert self._equivalent(cfg, reloaded)

    def test_yaml_round_trip(self, tmp_path):
        """A config written to YAML and loaded back matches the original."""
        import yaml

        cfg = TrainScopeConfig(
            run_dir="./runs",
            run_name="yaml_rt",
            detector={"name": "changepoint", "threshold": 6.0},
            activation_layer_filter=["attn", "mlp"],
            metric_plugins=["a.b", "c.d"],
            compaction_every_n_steps=250,
        )
        path = tmp_path / "config.yaml"
        path.write_text(yaml.safe_dump(cfg.to_dict()))
        reloaded = TrainScopeConfig.from_yaml(path)
        assert self._equivalent(cfg, reloaded)

    def test_json_round_trip(self, tmp_path):
        """load_config also accepts JSON files."""
        import json

        cfg = TrainScopeConfig(
            run_dir="./runs",
            run_name="json_rt",
            detector={"name": "percentile", "threshold": 0.99},
        )
        path = tmp_path / "config.json"
        path.write_text(json.dumps(cfg.to_dict()))
        reloaded = load_config(path)
        assert self._equivalent(cfg, reloaded)

    def test_env_round_trip(self, monkeypatch):
        """Every scalar config field, expressed as a TRAINSCOPE_* env var,
        loads to the same value as passing the equivalent dict directly."""
        overrides = {
            "TRAINSCOPE_FULL_RESOLUTION_WINDOW": "200",
            "TRAINSCOPE_DECIMATION_FACTOR": "20",
            "TRAINSCOPE_HISTOGRAM_EVERY_N_STEPS": "25",
            "TRAINSCOPE_ACTIVATION_METRICS_EVERY_N_STEPS": "3",
            "TRAINSCOPE_COMPACTION_EVERY_N_STEPS": "50",
            "TRAINSCOPE_TRACK_MEMORY": "false",
            "TRAINSCOPE_STOP_ON_SPIKE": "true",
            "TRAINSCOPE_RNG_EVERY_N_STEPS": "4",
            "TRAINSCOPE_RESUME": "true",
            "TRAINSCOPE_ACTIVATION_LAYER_FILTER": "[attn, mlp]",
            "TRAINSCOPE_DETECTOR": '{"name": "z_score", "threshold": 3.5}',
            "TRAINSCOPE_CHECKPOINT_ON_SPIKE": "{step}.pt",
            "TRAINSCOPE_ALERTS": '[{"type": "slack", "url": "https://x"}]',
        }
        monkeypatch.setenv("TRAINSCOPE_RUN_NAME", "env_rt")
        for key, value in overrides.items():
            monkeypatch.setenv(key, value)

        env_cfg = TrainScopeConfig.from_env()
        dict_cfg = TrainScopeConfig(
            run_name="env_rt",
            full_resolution_window=200,
            decimation_factor=20,
            histogram_every_n_steps=25,
            activation_metrics_every_n_steps=3,
            compaction_every_n_steps=50,
            track_memory=False,
            stop_on_spike=True,
            rng_every_n_steps=4,
            resume=True,
            activation_layer_filter=["attn", "mlp"],
            detector={"name": "z_score", "threshold": 3.5},
            checkpoint_on_spike="{step}.pt",
            alerts=[{"type": "slack", "url": "https://x"}],
        )
        assert self._equivalent(env_cfg, dict_cfg)

    def test_env_boolean_coercion(self, monkeypatch):
        """Boolean env vars accept the documented truthy/falsy spellings."""
        for raw, expected in [
            ("1", True),
            ("true", True),
            ("on", True),
            ("0", False),
            ("no", False),
        ]:
            monkeypatch.setenv("TRAINSCOPE_TRACK_MEMORY", raw)
            assert TrainScopeConfig.from_env().track_memory is expected

    def test_env_unknown_variables_ignored(self, monkeypatch):
        """Unknown TRAINSCOPE_* vars are skipped, not fatal (forward compat
        with config fields added in a newer minor release)."""
        monkeypatch.setenv("TRAINSCOPE_SOME_FUTURE_FIELD", "42")
        cfg = TrainScopeConfig.from_env()
        assert cfg.to_dict()["track_memory"] is True

    def test_detector_thresholds_reach_each_detector(self):
        """Per-detector thresholds survive the config round trip and reach
        the constructed detector."""
        from trainscope.core.detectors import make_detector

        cases = [
            ("z_score", {"threshold": 3.5}, "threshold"),
            ("changepoint", {"threshold": 6.0}, "threshold"),
            ("expert_utilization_drift", {"threshold": 0.85}, "threshold"),
            ("addressor_concentration_drift", {"threshold": 0.6}, "threshold"),
            ("percentile", {"lower": 5.0, "upper": 95.0}, "lower"),
        ]
        for name, kwargs, attr in cases:
            cfg = TrainScopeConfig(detector={"name": name, **kwargs})
            reloaded = load_config(cfg.to_dict())
            assert reloaded.detector == cfg.detector
            det = make_detector(reloaded)
            assert getattr(det, attr) == kwargs[attr]
