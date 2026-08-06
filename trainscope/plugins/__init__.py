"""Plugin system for trainscope metrics and anomaly detectors."""

from abc import ABC, abstractmethod
from importlib import import_module
from importlib.metadata import entry_points
from typing import Any, ClassVar, TypeVar

from trainscope.core.detectors import AnomalyDetector, register_detector

T = TypeVar("T")


class MetricPlugin(ABC):
    """Base class for custom global metric plugins.

    Plugins are discovered via the ``trainscope.plugins.metrics`` entry point
    and can also be loaded explicitly through
    :attr:`TrainScopeConfig.metric_plugins`.
    """

    name: ClassVar[str] = ""

    @abstractmethod
    def compute(self, model: Any, optimizer: Any, step: int) -> dict[str, float]:
        """Return a mapping from metric name to scalar value for ``step``."""


class AnomalyDetectorPlugin(AnomalyDetector):
    """Base class for custom anomaly-detector plugins.

    Plugins are discovered via the ``trainscope.plugins.detectors`` entry point
    and can also be loaded explicitly through
    :attr:`TrainScopeConfig.detector_plugins`.
    """

    name: ClassVar[str] = ""

    def __init__(self, **kwargs: Any):
        pass

    @abstractmethod
    def update(self, loss: float) -> float | None:
        """Incorporate ``loss`` and return an anomaly score if detected."""

    @property
    @abstractmethod
    def warmup(self) -> bool:
        """True when the detector is still warming up."""


def _load_class(dotted_path: str) -> type:
    if "." not in dotted_path:
        raise ValueError(f"Plugin class path must be dotted, got {dotted_path!r}")
    module_name, class_name = dotted_path.rsplit(".", 1)
    module = import_module(module_name)
    if not hasattr(module, class_name):
        raise ImportError(f"Module {module_name} has no attribute {class_name}")
    cls: type = getattr(module, class_name)
    return cls


def _entry_point_classes(group_name: str, base_cls: Any) -> list[Any]:
    classes: list[Any] = []
    try:
        eps: Any = entry_points()
        try:
            group = eps.select(group=group_name)
        except AttributeError:
            group = getattr(eps, "get", lambda k, default=None: [])(group_name, [])
    except Exception:
        return classes

    for ep in group:
        try:
            cls = ep.load()
            if issubclass(cls, base_cls):
                classes.append(cls)
        except Exception:
            # Malformed entry points are ignored so they do not break discovery.
            continue
    return classes


def discover_metric_plugins() -> list[type[MetricPlugin]]:
    """Return metric plugin classes declared via entry points."""
    return _entry_point_classes("trainscope.plugins.metrics", MetricPlugin)


def discover_detector_plugins() -> list[type[AnomalyDetectorPlugin]]:
    """Return detector plugin classes declared via entry points."""
    return _entry_point_classes("trainscope.plugins.detectors", AnomalyDetectorPlugin)


def load_metric_plugins(configured: list[str] | None = None) -> dict[str, type[MetricPlugin]]:
    """Load metric plugins from entry points and explicit config paths."""
    by_name: dict[str, type[MetricPlugin]] = {}
    for cls in discover_metric_plugins():
        by_name[cls.name or cls.__name__] = cls
    for dotted in configured or []:
        cls = _load_class(dotted)
        if not issubclass(cls, MetricPlugin):
            raise TypeError(f"{cls} is not a MetricPlugin")
        by_name[cls.name or cls.__name__] = cls
    return by_name


def load_detector_plugins(
    configured: list[str] | None = None,
) -> dict[str, type[AnomalyDetectorPlugin]]:
    """Load detector plugins from entry points and explicit config paths.

    Loaded detector plugins are registered in the global detector registry so
    they can be selected by name in :attr:`TrainScopeConfig.detector`.
    """
    by_name: dict[str, type[AnomalyDetectorPlugin]] = {}
    for cls in discover_detector_plugins():
        name = cls.name or cls.__name__
        by_name[name] = cls
        register_detector(name, cls)
    for dotted in configured or []:
        cls = _load_class(dotted)
        if not issubclass(cls, AnomalyDetector):
            raise TypeError(f"{cls} is not an AnomalyDetector")
        name = cls.name or cls.__name__
        by_name[name] = cls
        register_detector(name, cls)
    return by_name


def instantiate_metric_plugins(
    configured: list[str] | None = None,
) -> list[MetricPlugin]:
    """Instantiate all configured and entry-point metric plugins."""
    plugins: list[MetricPlugin] = []
    seen: set[str] = set()
    for cls in discover_metric_plugins():
        name = cls.name or cls.__name__
        if name not in seen:
            plugins.append(cls())
            seen.add(name)
    for dotted in configured or []:
        cls = _load_class(dotted)
        if not issubclass(cls, MetricPlugin):
            raise TypeError(f"{cls} is not a MetricPlugin")
        name = cls.name or cls.__name__
        if name not in seen:
            plugins.append(cls())
            seen.add(name)
    return plugins


__all__ = [
    "MetricPlugin",
    "AnomalyDetectorPlugin",
    "discover_metric_plugins",
    "discover_detector_plugins",
    "load_metric_plugins",
    "load_detector_plugins",
    "instantiate_metric_plugins",
]
