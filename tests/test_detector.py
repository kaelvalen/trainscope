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
