import collections


class RollingBuffer:
    """Stores training snapshots with a full-resolution tail and decimated history.

    The most recent ``full_resolution_window`` steps are kept at full resolution.
    When the full-resolution buffer overflows, the evicted step is retained in the
    decimated buffer iff ``step % decimation_factor == 0``. The decimated buffer is
    bounded to ``full_resolution_window * 10`` entries so disk/memory usage stays
    predictable.
    """

    def __init__(self, full_resolution_window: int = 500, decimation_factor: int = 10):
        if full_resolution_window < 1:
            raise ValueError("full_resolution_window must be >= 1")
        if decimation_factor < 1:
            raise ValueError("decimation_factor must be >= 1")

        self._full_resolution_window = full_resolution_window
        self._decimation_factor = decimation_factor
        self._full_resolution: collections.deque[dict] = collections.deque(
            maxlen=full_resolution_window
        )
        self._decimated: collections.deque[dict] = collections.deque(
            maxlen=full_resolution_window * 10
        )

    def add(self, global_snap: dict, layer_snaps: dict[str, dict]):
        step = global_snap.get("step")
        if step is None:
            raise ValueError("global_snap must contain a 'step' key")

        entry = {
            "global": global_snap,
            "layers": layer_snaps,
        }

        # Evict the oldest full-resolution entry to decimated history, keeping
        # every decimation_factor-th step.
        if len(self._full_resolution) == self._full_resolution_window:
            oldest = self._full_resolution.popleft()
            oldest_step = oldest["global"].get("step", 0)
            if oldest_step % self._decimation_factor == 0:
                self._decimated.append(oldest)

        self._full_resolution.append(entry)

    def get_global_steps(self) -> list[dict]:
        seen_steps = set()
        result = []
        for entry in self._decimated:
            s = entry["global"].get("step")
            if s not in seen_steps:
                seen_steps.add(s)
                result.append(entry["global"])
        for entry in self._full_resolution:
            s = entry["global"].get("step")
            if s not in seen_steps:
                seen_steps.add(s)
                result.append(entry["global"])
        result.sort(key=lambda x: x.get("step", 0))
        return result

    def get_layer_steps(self, layer_name: str) -> list[dict]:
        seen_steps = set()
        result = []
        for entry in self._decimated:
            s = entry["global"].get("step")
            if s not in seen_steps and layer_name in entry["layers"]:
                seen_steps.add(s)
                result.append(entry["layers"][layer_name])
        for entry in self._full_resolution:
            s = entry["global"].get("step")
            if s not in seen_steps and layer_name in entry["layers"]:
                seen_steps.add(s)
                result.append(entry["layers"][layer_name])
        result.sort(key=lambda x: x.get("step", 0))
        return result

    def get_window(self, center_step: int, before: int, after: int) -> list[dict]:
        lo, hi = center_step - before, center_step + after
        result = [
            entry
            for entry in list(self._decimated) + list(self._full_resolution)
            if lo <= entry["global"].get("step", 0) <= hi
        ]
        result.sort(key=lambda x: x["global"].get("step", 0))
        return result
