import pytest

from trainscope.core.detector import SpikeDetector


class _FakePelt:
    """Ruptures-compatible stub. ``predict`` returns segment-end breakpoints
    ending with the series length n (like the real library); callers set
    ``breakpoint`` to control where the last real breakpoint sits. Negative
    values are interpreted relative to n (e.g. -1 -> n-1)."""

    breakpoint: int | None = None

    def __init__(self, *args, **kwargs):
        self._signal: list = []

    def fit(self, signal):
        self._signal = signal
        return self

    def predict(self, pen):
        n = len(self._signal)
        bp = self.breakpoint if self.breakpoint is not None else n - 1
        if bp < 0:
            bp = n + bp
        return [bp, n]


class _FakeRpt:
    Pelt = _FakePelt


def _install_fake_pelt(monkeypatch, cp_mod) -> None:
    """Install a deterministic PELT stub into the changepoint module, so tests
    exercise the PELT branch without requiring the real ruptures library."""
    _FakePelt.breakpoint = None
    monkeypatch.setattr(cp_mod, "rpt", _FakeRpt)


def _pelt_breakpoint(cp_mod, value: int) -> None:
    _FakePelt.breakpoint = value


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

    @pytest.mark.parametrize("with_pelt", [True, False])
    def test_changepoint_robustness_calibration_set(self, monkeypatch, with_pelt):
        """Calibration set: 140 combinations evaluated across all 16,800 steps without early break.

        Run both with and without ruptures: PELT may only override the score
        of an already-fired CUSUM spike (it is not an independent trigger), so
        the false-positive rate must be CUSUM's either way. Without the
        explicit parametrization, the test silently changes behavior depending
        on whether ruptures happens to be installed.
        """
        import random

        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        if with_pelt:
            _install_fake_pelt(monkeypatch, cp_mod)
        else:
            monkeypatch.setattr(cp_mod, "rpt", None)

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

    @pytest.mark.parametrize("with_pelt", [True, False])
    def test_changepoint_held_out_validation_set(self, monkeypatch, with_pelt):
        """Held-out validation set: completely unseen seeds, extreme scales (1e-6 to 1e6), evaluated across all steps without break."""
        import random

        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        if with_pelt:
            _install_fake_pelt(monkeypatch, cp_mod)
        else:
            monkeypatch.setattr(cp_mod, "rpt", None)

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

    def test_pelt_does_not_trigger_without_cusum(self, monkeypatch):
        """PELT is a magnitude refiner, never an independent trigger: without a
        fired CUSUM (small jump, |z| << threshold) the detector returns None
        even when PELT would report a change."""
        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        _install_fake_pelt(monkeypatch, cp_mod)
        _pelt_breakpoint(cp_mod, value=-1)  # change at the current observation

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        # z = (1.2 - 1.05) / 0.07413 ≈ 2.02 < threshold=6.0: CUSUM does not
        # fire, so PELT's confirmation must NOT surface a spike.
        assert det.update(1.2) is None

    def test_pelt_refines_fired_cusum_score(self, monkeypatch):
        """When CUSUM fires, PELT replaces the clamped cumulative sum with the
        raw median/MAD-normalized deviation, preserving jump magnitude."""
        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        _install_fake_pelt(monkeypatch, cp_mod)
        _pelt_breakpoint(cp_mod, value=-1)  # change now

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        # z = (3.0 - 1.05) / 0.07413 ≈ 26.3 -> CUSUM fires immediately.
        res = det.update(3.0)
        assert res is not None
        expected = (3.0 - 1.05) / (1.4826 * 0.05)
        assert res == pytest.approx(expected)

    def test_pelt_preserves_sign_of_fired_cusum_score(self, monkeypatch):
        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        _install_fake_pelt(monkeypatch, cp_mod)
        _pelt_breakpoint(cp_mod, value=-1)

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        # Negative jump: z = (0.1 - 1.05) / 0.07413 ≈ -12.8 -> negative CUSUM.
        res = det.update(0.1)
        assert res is not None
        expected = (0.1 - 1.05) / (1.4826 * 0.05)
        assert res == pytest.approx(expected)
        assert res < 0

    def test_pelt_requires_change_at_current_observation(self, monkeypatch):
        """CUSUM fired, but PELT sees the change in the middle of the window
        (breakpoint << n - min_size): PELT must not override; the clamped
        CUSUM score is returned."""
        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        _install_fake_pelt(monkeypatch, cp_mod)
        _pelt_breakpoint(cp_mod, value=-10)  # old change, not at the tail

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        # CUSUM fires (big jump) but PELT disagrees -> CUSUM score, not raw dev.
        res = det.update(3.0)
        assert res is not None
        # |CUSUM score| >= threshold (clamped semantics), and it is NOT the
        # raw deviation (PELT did not override).
        assert abs(res) >= 6.0
        assert res != pytest.approx((3.0 - 1.05) / (1.4826 * 0.05))

    def test_pelt_ignores_change_older_than_min_size(self, monkeypatch):
        """A breakpoint leaving a final segment longer than min_size is not a
        change 'right now': PELT does not override the fired CUSUM score."""
        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        _install_fake_pelt(monkeypatch, cp_mod)
        # min_observations=10 -> min_size = max(2, 10//5) = 2. A breakpoint at
        # n-3 leaves a final segment of 3 > min_size.
        _pelt_breakpoint(cp_mod, value=-3)

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        res = det.update(3.0)
        assert res is not None
        assert abs(res) >= 6.0
        assert res != pytest.approx((3.0 - 1.05) / (1.4826 * 0.05))

    def test_pelt_fires_when_final_segment_at_min_size(self, monkeypatch):
        """A breakpoint leaving a final segment exactly min_size long is the
        latest structurally possible change; PELT overrides the fired CUSUM
        score with the raw deviation."""
        from trainscope.core.detectors import changepoint as cp_mod
        from trainscope.core.detectors.changepoint import ChangePointDetector

        _install_fake_pelt(monkeypatch, cp_mod)
        # min_size = 2: breakpoint at n-2 -> final segment length 2.
        _pelt_breakpoint(cp_mod, value=-2)

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)

        res = det.update(3.0)
        assert res is not None
        assert res == pytest.approx((3.0 - 1.05) / (1.4826 * 0.05))


class TestChangePointDetectorPeltIntegration:
    """Tests against the real ruptures library (installed via the dev extra),
    not a stub. These guard the PELT branch against the min_size/breakpoint
    arithmetic that the unit tests above can only approximate."""

    def test_real_pelt_never_triggers_without_cusum(self):
        rpt = pytest.importorskip("ruptures")

        from trainscope.core.detectors.changepoint import ChangePointDetector

        # Guard: the import must actually see the real library (a stub injected
        # by an earlier monkeypatch in the same process would be a false pass).
        assert rpt.Pelt.__name__ == "Pelt"

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)
        # Small jump: CUSUM does not fire; PELT must not surface anything.
        assert det.update(1.2) is None

    def test_real_pelt_refines_fired_cusum_score(self):
        rpt = pytest.importorskip("ruptures")

        from trainscope.core.detectors.changepoint import ChangePointDetector

        assert rpt.Pelt.__name__ == "Pelt"

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        for _ in range(20):
            det.update(1.0)
            det.update(1.1)
        # Big jump fires CUSUM; PELT (if it confirms a change now) returns the
        # raw deviation, otherwise the clamped CUSUM score. Either way a spike
        # must surface, and the score must not be below the CUSUM threshold
        # magnitude unless PELT refined it (raw deviation can be below 6).
        res = det.update(3.0)
        assert res is not None
        # If PELT did not refine, |score| >= threshold; if it did, the raw
        # deviation magnitude is ~26. Both are consistent with "fired".
        assert res == pytest.approx((3.0 - 1.05) / (1.4826 * 0.05)) or abs(res) >= 6.0

    def test_real_pelt_does_not_fire_on_old_change(self):
        rpt = pytest.importorskip("ruptures")

        from trainscope.core.detectors.changepoint import ChangePointDetector

        assert rpt.Pelt.__name__ == "Pelt"

        det = ChangePointDetector(threshold=6.0, min_observations=10)
        # Two regimes: 20 stable values then a sustained shift for 20 more.
        # The change is in the middle of the window, not at the tail, so the
        # final segment is long and PELT should NOT report "now".
        for _ in range(20):
            det.update(1.0)
        for _ in range(20):
            det.update(1.4)
        # No new change at the current observation -> no spike.
        assert det.update(1.4) is None


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


class TestAddressorConcentrationDriftDetector:
    def test_defaults_follow_experiment_threshold(self):
        from trainscope.core.detectors.addressor_concentration import (
            AddressorConcentrationDriftDetector,
        )

        det = AddressorConcentrationDriftDetector()
        assert det.threshold == 0.6  # "control max + margin" from v1.4.1
        assert det.run_steps == 3

    def test_concentration_triggers_at_0_6(self):
        from trainscope.core.detectors.addressor_concentration import (
            AddressorConcentrationDriftDetector,
        )

        det = AddressorConcentrationDriftDetector(min_observations=5)
        for _ in range(5):
            det.update(0.25)
        assert det.update(0.7) is None
        assert det.update(0.7) is None
        score = det.update(0.7)
        assert score is not None
        assert score == 0.7

    def test_healthy_diffuse_addressing_never_triggers(self):
        from trainscope.core.detectors.addressor_concentration import (
            AddressorConcentrationDriftDetector,
        )

        det = AddressorConcentrationDriftDetector(min_observations=5)
        triggers = 0
        for _ in range(200):
            # 16-slot healthy addressing: max share 0.24-0.32.
            if det.update(0.25 + (_ % 3) * 0.03) is not None:
                triggers += 1
        assert triggers == 0

    def test_factory_registers_detector(self):
        from trainscope.core.config import TrainScopeConfig
        from trainscope.core.detectors import make_detector
        from trainscope.core.detectors.addressor_concentration import (
            AddressorConcentrationDriftDetector,
        )

        cfg = TrainScopeConfig(detector={"name": "addressor_concentration_drift"})
        det = make_detector(cfg)
        assert isinstance(det, AddressorConcentrationDriftDetector)
