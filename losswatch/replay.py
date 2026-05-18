from typing import Iterator


class SkippingDataLoader:
    """Wraps any iterable DataLoader, yielding batches whose index is not in skip_batches.

    Usage::

        from losswatch.replay import SkippingDataLoader
        import json

        with open("replay_config.json") as f:
            cfg = json.load(f)

        loader = SkippingDataLoader(original_loader, skip_batches=cfg["skip_batches"])
        for batch in loader:
            loss = model(batch)
            ...
    """

    def __init__(self, loader, skip_batches: list[int]):
        self._loader = loader
        self._skip = set(skip_batches)

    def __iter__(self) -> Iterator:
        for i, batch in enumerate(self._loader):
            if i not in self._skip:
                yield batch

    def __len__(self) -> int:
        base = len(self._loader)
        skipped = sum(1 for i in self._skip if i < base)
        return max(0, base - skipped)
