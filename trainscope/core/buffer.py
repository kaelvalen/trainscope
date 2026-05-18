import collections


class RollingBuffer:
    def __init__(self, full_resolution_window: int = 500, decimation_factor: int = 10):
        self._full_resolution_window = full_resolution_window
        self._decimation_factor = decimation_factor
        self._full_resolution: collections.deque = collections.deque(maxlen=full_resolution_window)
        self._decimated: list[dict] = []
        self._step_count = 0
        self._eviction_buffer: collections.deque = collections.deque(maxlen=full_resolution_window)

    def add(self, global_snap: dict, layer_snaps: dict[str, dict]):
        combined = {
            "global": global_snap,
            "layers": layer_snaps,
            "step_number": self._step_count,
        }

        if len(self._full_resolution) == self._full_resolution_window:
            oldest = self._full_resolution[0]
            oldest_step = oldest["global"].get("step", oldest.get("step_number", 0))
            if oldest_step % self._decimation_factor == 0:
                self._decimated.append(oldest)

        self._full_resolution.append(combined)
        self._step_count += 1

    def get_global_steps(self) -> list[dict]:
        seen_steps = set()
        result = []
        for entry in self._decimated:
            s = entry["global"].get("step", entry.get("step_number"))
            if s not in seen_steps:
                seen_steps.add(s)
                result.append(entry["global"])
        for entry in self._full_resolution:
            s = entry["global"].get("step", entry.get("step_number"))
            if s not in seen_steps:
                seen_steps.add(s)
                result.append(entry["global"])
        result.sort(key=lambda x: x.get("step", 0))
        return result

    def get_layer_steps(self, layer_name: str) -> list[dict]:
        seen_steps = set()
        result = []
        for entry in self._decimated:
            s = entry["global"].get("step", entry.get("step_number"))
            if s not in seen_steps and layer_name in entry["layers"]:
                seen_steps.add(s)
                result.append(entry["layers"][layer_name])
        for entry in self._full_resolution:
            s = entry["global"].get("step", entry.get("step_number"))
            if s not in seen_steps and layer_name in entry["layers"]:
                seen_steps.add(s)
                result.append(entry["layers"][layer_name])
        result.sort(key=lambda x: x.get("step", 0))
        return result

    def get_window(self, center_step: int, before: int, after: int) -> list[dict]:
        all_entries = list(self._decimated) + list(self._full_resolution)
        result = []
        for entry in all_entries:
            s = entry["global"].get("step", entry.get("step_number", 0))
            if (center_step - before) <= s <= (center_step + after):
                result.append(entry)
        result.sort(key=lambda x: x["global"].get("step", x.get("step_number", 0)))
        return result
