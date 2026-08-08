import pytest

from trainscope.core.buffer import RollingBuffer


def make_snap(step: int):
    return {"step": step, "loss": float(step) * 0.1}


def make_layer_snap(step: int, layer: str = "layer0"):
    return {layer: {"step": step, "grad_l2_norm": 0.5}}


class TestRollingBuffer:
    def test_add_and_get_global(self):
        buf = RollingBuffer(full_resolution_window=10, decimation_factor=5)
        for i in range(5):
            buf.add(make_snap(i), make_layer_snap(i))
        steps = buf.get_global_steps()
        assert len(steps) == 5
        assert steps[0]["step"] == 0
        assert steps[-1]["step"] == 4

    def test_full_resolution_fills(self):
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(5):
            buf.add(make_snap(i), make_layer_snap(i))
        assert len(buf._full_resolution) == 5

    def test_decimation_on_overflow(self):
        window = 5
        factor = 2
        buf = RollingBuffer(full_resolution_window=window, decimation_factor=factor)
        for i in range(window + 1):
            buf.add(make_snap(i), make_layer_snap(i))
        # step 0 should be evicted; if 0 % 2 == 0 it goes to decimated
        assert len(buf._decimated) >= 1
        decimated_steps = [e["global"]["step"] for e in buf._decimated]
        assert 0 in decimated_steps

    def test_decimation_skips_non_multiple(self):
        window = 5
        factor = 5
        buf = RollingBuffer(full_resolution_window=window, decimation_factor=factor)
        # add window+1 steps; step 0 is evicted, 0 % 5 == 0 → goes to decimated
        for i in range(window + 1):
            buf.add(make_snap(i), make_layer_snap(i))
        decimated_steps = [e["global"]["step"] for e in buf._decimated]
        for s in decimated_steps:
            assert s % factor == 0

    def test_get_layer_steps(self):
        buf = RollingBuffer(full_resolution_window=10, decimation_factor=5)
        for i in range(6):
            buf.add(make_snap(i), {"layer_A": {"step": i, "grad_l2_norm": float(i)}})
        layer_steps = buf.get_layer_steps("layer_A")
        assert all("grad_l2_norm" in s for s in layer_steps)

    def test_get_layer_steps_missing_layer(self):
        buf = RollingBuffer(full_resolution_window=10, decimation_factor=5)
        for i in range(3):
            buf.add(make_snap(i), make_layer_snap(i))
        result = buf.get_layer_steps("nonexistent")
        assert result == []

    def test_get_window_correct_slice(self):
        buf = RollingBuffer(full_resolution_window=100, decimation_factor=10)
        for i in range(100):
            buf.add(make_snap(i), make_layer_snap(i))
        window = buf.get_window(center_step=50, before=5, after=5)
        win_steps = [e["global"]["step"] for e in window]
        assert 50 in win_steps
        assert all(45 <= s <= 55 for s in win_steps)

    def test_get_window_boundary(self):
        buf = RollingBuffer(full_resolution_window=100, decimation_factor=10)
        for i in range(20):
            buf.add(make_snap(i), make_layer_snap(i))
        window = buf.get_window(center_step=0, before=5, after=5)
        win_steps = [e["global"]["step"] for e in window]
        assert 0 in win_steps

    def test_chronological_order(self):
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(10):
            buf.add(make_snap(i), make_layer_snap(i))
        steps = [s["step"] for s in buf.get_global_steps()]
        assert steps == sorted(steps)

    def test_no_duplicate_steps(self):
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(12):
            buf.add(make_snap(i), make_layer_snap(i))
        steps = [s["step"] for s in buf.get_global_steps()]
        assert len(steps) == len(set(steps))

    def test_decimated_history_bounded(self):
        window = 10
        factor = 2
        buf = RollingBuffer(full_resolution_window=window, decimation_factor=factor)
        for i in range(window * 100):
            buf.add(make_snap(i), make_layer_snap(i))
        assert len(buf._decimated) <= window * 10

    def test_requires_step_key(self):
        buf = RollingBuffer()
        with pytest.raises(ValueError):
            buf.add({"loss": 1.0}, {})

    def test_get_global_steps_sorted_across_segments(self):
        """Chronological order must hold across the decimated/full-resolution
        boundary: all decimated steps precede all full-resolution steps."""
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(25):
            buf.add(make_snap(i), make_layer_snap(i))

        assert len(buf._decimated) > 0
        assert len(buf._full_resolution) == 5
        decimated_steps = [e["global"]["step"] for e in buf._decimated]
        full_steps = [e["global"]["step"] for e in buf._full_resolution]
        assert max(decimated_steps) < min(full_steps)

        steps = [s["step"] for s in buf.get_global_steps()]
        assert steps == sorted(steps)
        assert len(steps) == len(set(steps))

    def test_get_layer_steps_sorted_across_segments(self):
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(20):
            buf.add(make_snap(i), {"layer_A": {"step": i, "grad_l2_norm": float(i)}})
        steps = [s["step"] for s in buf.get_layer_steps("layer_A")]
        assert steps == sorted(steps)

    def test_get_window_sorted_across_segments(self):
        """A window spanning the segment boundary must stay sorted."""
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(25):
            buf.add(make_snap(i), make_layer_snap(i))
        window = buf.get_window(center_step=12, before=8, after=8)
        win_steps = [e["global"]["step"] for e in window]
        assert win_steps == sorted(win_steps)
        assert len(win_steps) == len(set(win_steps))

    def test_merge_falls_back_to_sort_on_unsorted_segments(self):
        """If a future insertion path ever violates the segment ordering
        invariant, getters must still return chronological rows instead of
        silently returning out-of-order data."""
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(10):
            buf.add(make_snap(i), make_layer_snap(i))
        # Simulate a violation: an out-of-order step appended to decimated.
        buf._decimated.append({"global": make_snap(99), "layers": {}})

        steps = [s["step"] for s in buf.get_global_steps()]
        assert steps == sorted(steps)
        assert 99 in steps

        window = buf.get_window(center_step=99, before=5, after=5)
        win_steps = [e["global"]["step"] for e in window]
        assert win_steps == sorted(win_steps)

    def test_get_window_includes_decimated_and_full(self):
        buf = RollingBuffer(full_resolution_window=5, decimation_factor=2)
        for i in range(20):
            buf.add(make_snap(i), make_layer_snap(i))
        window = buf.get_window(center_step=2, before=5, after=5)
        win_steps = [e["global"]["step"] for e in window]
        assert 2 in win_steps
        assert all(0 <= s <= 7 for s in win_steps)
