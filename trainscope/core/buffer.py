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

    @staticmethod
    def _merged_values(combined: dict[int, dict], key) -> list[dict]:
        """Return merged values in chronological order.

        Fast path: both segments are step-sorted and disjoint (eviction moves
        a step from full resolution to decimated history exactly once, so
        decimated holds strictly older steps), making insertion order already
        chronological — no sort needed. A future insertion path that violates
        this invariant would otherwise silently return out-of-order rows, so
        an O(n) order check falls back to an explicit sort on violation.
        """
        result = list(combined.values())
        if any(key(a) > key(b) for a, b in zip(result, result[1:])):
            result.sort(key=key)
        return result

    def get_global_steps(self) -> list[dict]:
        # Both deques are step-sorted and disjoint (eviction moves a step from
        # full resolution to decimated history exactly once), so a linear
        # dedup-merge preserves chronological order without sorting.
        combined: dict[int, dict] = {}
        for entry in self._decimated:
            combined[entry["global"]["step"]] = entry["global"]
        for entry in self._full_resolution:
            combined[entry["global"]["step"]] = entry["global"]
        return self._merged_values(combined, lambda s: s["step"])

    def get_layer_steps(self, layer_name: str) -> list[dict]:
        combined: dict[int, dict] = {}
        for entry in self._decimated:
            step = entry["global"]["step"]
            if step not in combined and layer_name in entry["layers"]:
                combined[step] = entry["layers"][layer_name]
        for entry in self._full_resolution:
            step = entry["global"]["step"]
            if step not in combined and layer_name in entry["layers"]:
                combined[step] = entry["layers"][layer_name]
        return self._merged_values(combined, lambda s: s["step"])

    def get_window(self, center_step: int, before: int, after: int) -> list[dict]:
        lo, hi = center_step - before, center_step + after
        combined: dict[int, dict] = {}
        for entry in self._decimated:
            step = entry["global"]["step"]
            if lo <= step <= hi:
                combined[step] = entry
        for entry in self._full_resolution:
            step = entry["global"]["step"]
            if lo <= step <= hi:
                combined[step] = entry
        return self._merged_values(combined, lambda e: e["global"]["step"])
