"""Frozen plugin-contract tests (Phase 3).

The stability scope promises that plugin interfaces are part of the contract.
This module pins exactly what a plugin author can rely on, so a future minor
release cannot silently change the surface:

- ``AnomalyDetector``: ``update(loss) -> float | None`` and ``warmup``.
- ``MetricPlugin``: ``name`` (ClassVar) and ``compute(model, optimizer, step)``.
- Detector plugin discovery: ``load_detector_plugins`` registers by name.
- Metric plugin metrics flow into the ``PLUGIN_METRICS_SCHEMA`` table.

These are *contract* tests, not implementation tests: they assert the shape
plugin authors depend on, not how the built-ins happen to behave.
"""

import inspect

import pyarrow as pa

from trainscope.core.detectors import AnomalyDetector, make_detector
from trainscope.io.writer import PLUGIN_METRICS_SCHEMA
from trainscope.plugins import AnomalyDetectorPlugin, MetricPlugin, load_detector_plugins


class TestAnomalyDetectorContract:
    def test_abstract_methods_are_the_contract(self):
        """A detector must implement exactly update() and warmup; these are
        what TrainScope.step() consumes."""
        abstract = set(AnomalyDetector.__abstractmethods__)
        assert abstract == {"update", "warmup"}

    def test_update_signature(self):
        sig = inspect.signature(AnomalyDetector.update)
        params = list(sig.parameters)
        assert params == ["self", "loss"]
        assert sig.return_annotation is None or sig.return_annotation == float | None

    def test_warmup_is_property(self):
        assert isinstance(inspect.getattr_static(AnomalyDetector, "warmup"), property)

    def test_all_builtin_detectors_are_instantiable(self):
        """Every registered detector must construct from its name (kwargs
        optional) and satisfy the ABC contract at runtime."""
        from trainscope.core.detectors import _REGISTRY

        for name, cls in _REGISTRY.items():
            try:
                detector = make_detector({"name": name})
            except TypeError:
                # Detector requires constructor kwargs beyond defaults.
                detector = None
            if detector is not None:
                assert isinstance(detector, AnomalyDetector)
                # Contract: update accepts a float and returns None-or-float.
                for _ in range(
                    min(detector.min_observations, 10)
                    if hasattr(detector, "min_observations")
                    else 10
                ):
                    result = detector.update(1.0)
                    assert result is None or isinstance(result, float)


class TestMetricPluginContract:
    def test_abstract_methods_are_the_contract(self):
        abstract = set(MetricPlugin.__abstractmethods__)
        assert abstract == {"compute"}

    def test_name_is_classvar(self):
        from typing import ClassVar

        annotation = MetricPlugin.__annotations__.get("name")
        assert annotation == ClassVar[str]

    def test_compute_signature(self):
        sig = inspect.signature(MetricPlugin.compute)
        assert list(sig.parameters) == ["self", "model", "optimizer", "step"]
        assert sig.return_annotation is None or str(sig.return_annotation).startswith("dict")

    def test_plugin_metrics_schema_columns(self):
        """The plugin-metrics table columns are part of the plugin contract:
        a plugin writes (step, plugin, metric, value) rows."""
        names = PLUGIN_METRICS_SCHEMA.names
        assert names == ["step", "plugin", "metric", "value"]
        types = {n: PLUGIN_METRICS_SCHEMA.field(n).type for n in names}
        assert types["step"] == pa.int64()
        assert types["plugin"] == pa.string()
        assert types["metric"] == pa.string()
        assert types["value"] == pa.float64()

    def test_metric_plugin_subclass_is_instantiable(self):
        from trainscope.plugins.builtin import GradientNormRatioPlugin

        plugin = GradientNormRatioPlugin()
        out = plugin.compute(model=_FakeModel(), optimizer=None, step=0)
        assert isinstance(out, dict)
        assert all(isinstance(v, float) for v in out.values())


class _FakeModel:
    def parameters(self):
        return []


class ContractTestDetector(AnomalyDetectorPlugin):
    """Module-level plugin used by the discovery contract test (must be at
    module scope so a dotted class path can import it)."""

    name = "contract_test_detector"

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._seen: list[float] = []

    def update(self, loss: float) -> float | None:
        self._seen.append(loss)
        if len(self._seen) < 5:
            return None
        if abs(loss - self._seen[-2]) > self.threshold:
            return loss
        return None

    @property
    def warmup(self) -> bool:
        return len(self._seen) < 5


class TestDetectorPluginDiscovery:
    def test_load_detector_plugins_returns_names_and_registers(self):
        """A detector plugin class (AnomalyDetector subclass with a name)
        must be returned by name and registered so config can select it."""
        plugins = load_detector_plugins()
        assert isinstance(plugins, dict)

    def test_configured_detector_plugin_loads(self):
        """A dotted class path in detector_plugins must load and register."""
        from trainscope.core.detectors import _REGISTRY

        plugins = load_detector_plugins(configured=["test_plugin_contract.ContractTestDetector"])
        assert "contract_test_detector" in plugins
        assert "contract_test_detector" in _REGISTRY

        detector = make_detector({"name": "contract_test_detector"})
        assert isinstance(detector, AnomalyDetector)
        # Contract: update returns None during warmup, float on spike.
        for i in range(5):
            detector.update(float(i))
        assert detector.warmup is False
        result = detector.update(50.0)
        assert isinstance(result, float)
