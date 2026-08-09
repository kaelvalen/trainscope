from trainscope.core.detector import SpikeDetector


class TestSpikeDetector:
    def test_first_30_updates_return_none(self):
        det = SpikeDetector(threshold=3.5)
        for i in range(30):
            result = det.update(1.0)
            assert result is None, f"Expected None at step {i}, got {result}"

    def test_stable_sequence_no_trigger(self):
        det = SpikeDetector(threshold=3.5)
        for i in range(100):
            result = det.update(1.0 + 0.001 * ((i % 3) - 1))
            if len(det._history) > 30:
                assert result is None

    def test_large_outlier_triggers(self):
        det = SpikeDetector(threshold=3.5)
        for _ in range(50):
            det.update(1.0)
        result = det.update(100.0)
        assert result is not None
        assert result > 3.5

    def test_negative_spike_triggers(self):
        det = SpikeDetector(threshold=3.5)
        for _ in range(50):
            det.update(10.0)
        result = det.update(-100.0)
        assert result is not None
        assert result < -3.5

    def test_z_score_approximately_correct(self):
        det = SpikeDetector(threshold=0.0)
        import random

        random.seed(42)
        base = [random.gauss(0, 1) for _ in range(100)]
        for v in base:
            det.update(v)
        result = det.update(1000.0)
        assert result is not None
        assert result > 5.0

    def test_threshold_respected(self):
        import random

        random.seed(0)
        det = SpikeDetector(threshold=10.0)
        for _ in range(50):
            det.update(1.0 + random.gauss(0, 0.1))
        # z ≈ (1.5 - 1.0) / 0.1 = 5, well below threshold 10
        result = det.update(1.5)
        assert result is None

    def test_spike_does_not_contaminate_baseline(self):
        # After a spike, subsequent stable values should not trigger.
        det = SpikeDetector(threshold=3.5, window=200)
        for _ in range(100):
            det.update(1.0)
        det.update(500.0)  # spike — must NOT shift mean enough to suppress next spike
        # A second spike of similar magnitude must still fire.
        result = det.update(500.0)
        assert result is not None

    def test_rolling_window_forgets_old_values(self):
        import random

        random.seed(1)
        # With window=50, values added more than 50 steps ago are forgotten.
        det = SpikeDetector(threshold=3.5, window=50)
        for _ in range(50):
            det.update(1000.0 + random.gauss(0, 1.0))  # high-loss phase
        # Switch to low-loss phase — window rolls over after 50 steps.
        for _ in range(50):
            det.update(1.0 + random.gauss(0, 0.01))  # std ~0.01
        # Baseline is now low-loss window; 1.01 is ~1σ away — no spike.
        result = det.update(1.01)
        assert result is None

    def test_returns_float_or_none(self):
        det = SpikeDetector(threshold=3.5)
        for _ in range(50):
            det.update(1.0)
        result = det.update(1.0)
        assert result is None or isinstance(result, float)

    def test_warmup_property(self):
        det = SpikeDetector(min_observations=10)
        assert det.warmup is True
        for _ in range(10):
            det.update(1.0)
        assert det.warmup is False

    def test_custom_min_observations(self):
        det = SpikeDetector(threshold=3.5, min_observations=5)
        for _ in range(5):
            det.update(1.0)
        result = det.update(100.0)
        assert result is not None

    def test_welford_matches_baseline_statistics(self):
        import random

        random.seed(7)
        det = SpikeDetector(threshold=3.5, window=50)
        values = [random.gauss(2.0, 0.5) for _ in range(50)]
        for v in values:
            det.update(v)
        # Welford mean/variance should equal the classic one over the history.
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        assert abs(det._mean - mean) < 1e-6
        assert abs(det._m2 / (det._count - 1) - var) < 1e-6

    def test_robust_mode_median_mad(self):
        det = SpikeDetector(threshold=3.5, min_observations=10, robust=True)
        for _ in range(10):
            det.update(1.0)
        result = det.update(100.0)
        assert result is not None
        assert result > 0

    def test_robust_mode_respects_threshold(self):
        det = SpikeDetector(threshold=10.0, min_observations=10, robust=True)
        for _ in range(5):
            det.update(1.0)
        for _ in range(5):
            det.update(1.1)  # introduce non-zero MAD
        # median=1.05, MAD=0.05 -> robust z for 1.5 is ~6, below threshold 10.
        result = det.update(1.5)
        assert result is None


class TestChangePointDetector:
    def test_changepoint_sudden_spike(self):
        from trainscope.core.detectors.changepoint import ChangePointDetector

        cp_det = ChangePointDetector(threshold=6.0, min_observations=30)
        for _ in range(40):
            cp_det.update(1.0)
        res = cp_det.update(10.0)
        assert res is not None
        assert res >= 6.0

    def test_changepoint_detects_slow_drift_where_zscore_misses(self):
        """Empirically prove CUSUM catches slow persistent loss drift earlier than Z-Score."""
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector
        from trainscope.core.detectors.z_score import ZScoreDetector

        random.seed(42)
        z_det = ZScoreDetector(threshold=3.5, min_observations=30)
        cp_det = ChangePointDetector(threshold=6.0, slack=1.0, min_observations=30)

        # Baseline warmup phase (50 steps around mean=1.0, std=0.1)
        for _ in range(50):
            val = 1.0 + random.gauss(0, 0.05)
            z_det.update(val)
            cp_det.update(val)

        # Inject slow cumulative drift (+0.01 per step, i.e., +0.2 std per step)
        z_step_detected = None
        cp_step_detected = None

        for step_i in range(1, 35):
            drift_val = 1.0 + 0.01 * step_i + random.gauss(0, 0.02)
            z_res = z_det.update(drift_val)
            cp_res = cp_det.update(drift_val)

            if z_res is not None and z_step_detected is None:
                z_step_detected = step_i
            if cp_res is not None and cp_step_detected is None:
                cp_step_detected = step_i

        # CUSUM must detect the slow drift
        assert cp_step_detected is not None, "ChangePointDetector failed to detect slow drift"
        # CUSUM must detect drift earlier than Z-Score (or Z-Score misses it entirely)
        assert z_step_detected is None or cp_step_detected < z_step_detected, (
            f"Expected CUSUM ({cp_step_detected}) to trigger earlier than ZScore ({z_step_detected})"
        )

    def test_changepoint_detects_accumulating_stress_before_loss_peak(self):
        """Detect a noisy, cumulative failure process without a scripted spike step."""
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector

        rng = random.Random("noisy-training-run")
        detector = ChangePointDetector(threshold=10.0, slack=1.0, min_observations=30)
        stress = 0.0
        losses: list[float] = []
        detection_steps: list[int] = []

        for step in range(180):
            # Model optimizer/data stress behaves like a noisy process: small
            # shocks accumulate, decay, and eventually amplify the loss.
            stress = max(0.0, stress * 0.985 + rng.gauss(0.012, 0.018))
            drift = max(0.0, stress - 0.18)
            loss = (1.0 + rng.gauss(0.0, 0.05)) * (
                1.0 + 0.25 * drift + 1.2 * drift**2 + 4.0 * max(0.0, drift - 0.45) ** 3
            )
            losses.append(loss)

            if detector.update(loss) is not None:
                detection_steps.append(step)

        assert detection_steps, "CUSUM failed to detect the accumulating stress"
        peak_step = max(range(len(losses)), key=losses.__getitem__)
        first_detection = detection_steps[0]
        assert first_detection < peak_step
        assert peak_step - first_detection >= 10

    def test_changepoint_sensitivity_across_scales(self):
        """Verify CUSUM sensitivity (true positive rate) across 5 different noise scales."""
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector

        scales = [1e-4, 0.01, 1.0, 100.0, 1e5]
        detected_count = 0

        for scale in scales:
            random.seed(42)
            cp_det = ChangePointDetector(threshold=6.0, slack=1.0, min_observations=30)
            # Warmup
            for _ in range(50):
                cp_det.update(random.gauss(0, scale))

            # Drift: +0.25 * scale per step
            detected = False
            for step_i in range(1, 40):
                val = 0.25 * scale * step_i + random.gauss(0, scale)
                if cp_det.update(val) is not None:
                    detected = True
                    break
            if detected:
                detected_count += 1

        assert detected_count == len(scales), (
            f"CUSUM failed to detect drift on some scales: {detected_count}/{len(scales)}"
        )

    def test_changepoint_sensitivity_by_drift_magnitude(self):
        """Verify CUSUM sensitivity across drift magnitudes from +0.10σ to +0.50σ per step."""
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector

        drift_rates = [0.10, 0.15, 0.25, 0.50]
        scales = [1e-3, 1.0, 1e3]

        for drift_rate in drift_rates:
            for scale in scales:
                rng = random.Random(f"drift_{drift_rate}_{scale}")
                cp_det = ChangePointDetector(threshold=6.0, slack=1.0, min_observations=30)
                # Warmup
                for _ in range(50):
                    cp_det.update(rng.gauss(0, scale))

                # Drift
                step_detected = None
                for step_i in range(1, 60):
                    val = drift_rate * scale * step_i + rng.gauss(0, scale)
                    if cp_det.update(val) is not None:
                        step_detected = step_i
                        break

                assert step_detected is not None, (
                    f"Failed to detect drift_rate={drift_rate} at scale={scale}"
                )

    def test_changepoint_no_false_positives_on_pure_noise(self):
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector

        random.seed(123)
        cp_det = ChangePointDetector(threshold=6.0, slack=1.0, min_observations=30)
        triggers = 0
        for _ in range(100):
            val = 1.0 + random.gauss(0, 0.1)
            res = cp_det.update(val)
            if res is not None:
                triggers += 1
        assert triggers == 0

    def test_changepoint_robustness_calibration_set(self):
        """Calibration set: 140 combinations evaluated across all 16,800 steps without early break."""
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector

        seeds = [1, 7, 42, 100, 123, 2024, 9999]
        scales = [0.01, 0.1, 1.0, 10.0, 100.0]
        means = [0.1, 1.0, 50.0, 1000.0]

        total_steps = 0
        false_positives = 0

        for seed in seeds:
            for scale in scales:
                for mean_val in means:
                    rng = random.Random(f"calib_{seed}_{scale}_{mean_val}")
                    cp_det = ChangePointDetector(threshold=6.0, slack=1.0, min_observations=30)
                    for _ in range(120):
                        total_steps += 1
                        val = mean_val + rng.gauss(0, scale)
                        if cp_det.update(val) is not None:
                            false_positives += 1

        fp_rate = false_positives / total_steps
        assert fp_rate <= 0.0005, (
            f"False positive step rate in calibration set exceeded 0.05%: {false_positives} / {total_steps} steps ({fp_rate:.4%})"
        )

    def test_changepoint_held_out_validation_set(self):
        """Held-out validation set: completely unseen seeds, extreme scales (1e-6 to 1e6), evaluated across all steps without break."""
        import random

        from trainscope.core.detectors.changepoint import ChangePointDetector

        held_out_seeds = [555, 777, 8888, 12345, 67890, 999999]
        extreme_scales = [1e-6, 1e-4, 0.05, 2.5, 500.0, 1e6]
        extreme_means = [1e-3, 0.5, 100.0, 1e6]

        total_steps = 0
        false_positives = 0

        for seed in held_out_seeds:
            for scale in extreme_scales:
                for mean_val in extreme_means:
                    rng = random.Random(f"heldout_{seed}_{scale}_{mean_val}")
                    cp_det = ChangePointDetector(threshold=6.0, slack=1.0, min_observations=30)
                    for _ in range(120):
                        total_steps += 1
                        val = mean_val + rng.gauss(0, scale)
                        if cp_det.update(val) is not None:
                            false_positives += 1

        fp_rate = false_positives / total_steps
        assert fp_rate <= 0.001, (
            f"False positive step rate on held-out set exceeded 0.1%: {false_positives} / {total_steps} steps ({fp_rate:.4%})"
        )

    def test_pelt_path_preserves_raw_deviation_magnitude(self, monkeypatch):
        """PELT-triggered spikes must return the raw normalized deviation, not
        a value clamped to the threshold: subtle change points (|dev| below
        threshold) must carry their true (small) magnitude instead of a flat
        |score| == threshold that makes every PELT spike look identical."""
        import pytest

        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        class _FakePelt:
            def __init__(self, *args, **kwargs):
                self._signal: list = []

            def fit(self, signal):
                self._signal = signal
                return self

            def predict(self, pen):
                # Always report a change point at the current observation.
                return [len(self._signal), len(self._signal)]

        class _FakeRpt:
            Pelt = _FakePelt

        monkeypatch.setattr(cp_mod, "rpt", _FakeRpt)

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        # Median 1.05, MAD 0.05 -> std = 1.4826 * 0.05 = 0.07413.
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        # dev = (1.2 - 1.05) / 0.07413 ≈ 2.02, well below threshold=6.0.
        res = det.update(1.2)
        assert res is not None
        expected = (1.2 - 1.05) / (1.4826 * 0.05)
        assert res == pytest.approx(expected)
        assert abs(res) < 6.0, "PELT score must not be clamped up to the threshold"

    def test_pelt_path_preserves_sign_and_magnitude(self, monkeypatch):
        import pytest

        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        class _FakePelt:
            def __init__(self, *args, **kwargs):
                self._signal: list = []

            def fit(self, signal):
                self._signal = signal
                return self

            def predict(self, pen):
                return [len(self._signal), len(self._signal)]

        class _FakeRpt:
            Pelt = _FakePelt

        monkeypatch.setattr(cp_mod, "rpt", _FakeRpt)

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        # Negative jump of the same magnitude as the positive test above.
        res = det.update(0.9)
        assert res is not None
        expected = (0.9 - 1.05) / (1.4826 * 0.05)
        assert res == pytest.approx(expected)
        assert res < 0
        assert abs(res) < 6.0


class TestExpertUtilizationDriftDetector:
    def test_warmup_requires_min_observations(self):
        from trainscope.core.detectors.expert_utilization import (
            ExpertUtilizationDriftDetector,
        )

        det = ExpertUtilizationDriftDetector(min_observations=5)
        assert det.warmup is True
        for _ in range(4):
            assert det.update(0.5) is None
        assert det.warmup is True
        # 5th observation exits warmup.
        assert det.update(0.5) is None
        assert det.warmup is False

    def test_concentration_durable_across_run_steps_triggers(self):
        from trainscope.core.detectors.expert_utilization import (
            ExpertUtilizationDriftDetector,
        )

        det = ExpertUtilizationDriftDetector(threshold=0.85, min_observations=5, run_steps=3)
        for _ in range(5):
            det.update(0.5)
        # Two consecutive high steps are not enough.
        assert det.update(0.9) is None
        assert det.update(0.9) is None
        # Third consecutive high step triggers with the concentration score.
        score = det.update(0.9)
        assert score is not None
        assert score == 0.9

    def test_drop_below_threshold_resets_run(self):
        from trainscope.core.detectors.expert_utilization import (
            ExpertUtilizationDriftDetector,
        )

        det = ExpertUtilizationDriftDetector(threshold=0.85, min_observations=5, run_steps=3)
        for _ in range(5):
            det.update(0.5)
        det.update(0.9)
        det.update(0.9)
        det.update(0.5)  # breaks the run
        assert det.update(0.9) is None  # run restarted
        assert det.update(0.9) is None
        assert det.update(0.9) is not None

    def test_no_false_positive_on_balanced_routing(self):
        from trainscope.core.detectors.expert_utilization import (
            ExpertUtilizationDriftDetector,
        )

        det = ExpertUtilizationDriftDetector(threshold=0.85, min_observations=5, run_steps=3)
        triggers = 0
        for _ in range(100):
            # Top-2-of-4 healthy routing keeps max share around 0.4-0.6.
            if det.update(0.5 + (_ % 3) * 0.05) is not None:
                triggers += 1
        assert triggers == 0

    def test_factory_registers_detector(self):
        from trainscope.core.config import TrainScopeConfig
        from trainscope.core.detectors import make_detector
        from trainscope.core.detectors.expert_utilization import (
            ExpertUtilizationDriftDetector,
        )

        cfg = TrainScopeConfig(detector={"name": "expert_utilization_drift", "threshold": 0.9})
        det = make_detector(cfg)
        assert isinstance(det, ExpertUtilizationDriftDetector)
        assert det.threshold == 0.9
