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
