import math

import torch

from losswatch.core.metrics import (
    compute_activation_metrics,
    compute_gradient_metrics,
    compute_weight_metrics,
    compute_weight_histogram,
)


class TestComputeActivationMetrics:
    def test_normal_tensor_finite(self):
        t = torch.randn(16, 32)
        m = compute_activation_metrics(t)
        assert all(math.isfinite(v) for v in m.values())
        assert set(m.keys()) == {"act_mean", "act_std", "act_max_abs", "act_kurtosis"}

    def test_empty_tensor(self):
        t = torch.empty(0)
        m = compute_activation_metrics(t)
        assert m["act_mean"] == 0.0
        assert m["act_std"] == 0.0
        assert m["act_max_abs"] == 0.0
        assert m["act_kurtosis"] == 0.0

    def test_all_zero_tensor(self):
        t = torch.zeros(8, 8)
        m = compute_activation_metrics(t)
        assert m["act_mean"] == 0.0
        assert m["act_std"] == 0.0
        assert m["act_kurtosis"] == 0.0

    def test_returns_python_float(self):
        t = torch.randn(4, 4)
        m = compute_activation_metrics(t)
        for v in m.values():
            assert isinstance(v, float)

    def test_kurtosis_gaussian_approx_zero(self):
        torch.manual_seed(0)
        t = torch.randn(10000)
        m = compute_activation_metrics(t)
        assert abs(m["act_kurtosis"]) < 0.5


class TestComputeGradientMetrics:
    def test_normal_grad(self):
        g = torch.randn(16, 32)
        m = compute_gradient_metrics(g)
        assert m["grad_l2_norm"] > 0.0
        assert m["grad_nan_inf_ratio"] == 0.0

    def test_none_grad(self):
        m = compute_gradient_metrics(None)
        assert m["grad_l2_norm"] == 0.0
        assert m["grad_nan_inf_ratio"] == 0.0

    def test_empty_grad(self):
        m = compute_gradient_metrics(torch.empty(0))
        assert m["grad_l2_norm"] == 0.0

    def test_nan_inf_ratio(self):
        g = torch.tensor([1.0, float("nan"), float("inf"), 2.0])
        m = compute_gradient_metrics(g)
        assert abs(m["grad_nan_inf_ratio"] - 0.5) < 1e-6

    def test_returns_python_float(self):
        g = torch.randn(4, 4)
        m = compute_gradient_metrics(g)
        for v in m.values():
            assert isinstance(v, float)


class TestComputeWeightMetrics:
    def test_normal_weight(self):
        w = torch.randn(16, 32)
        m = compute_weight_metrics(w)
        assert m["weight_l2_norm"] > 0.0

    def test_empty_weight(self):
        m = compute_weight_metrics(torch.empty(0))
        assert m["weight_l2_norm"] == 0.0

    def test_known_norm(self):
        w = torch.ones(3, 4)
        m = compute_weight_metrics(w)
        expected = math.sqrt(12)
        assert abs(m["weight_l2_norm"] - expected) < 1e-4

    def test_returns_python_float(self):
        w = torch.randn(4, 4)
        m = compute_weight_metrics(w)
        assert isinstance(m["weight_l2_norm"], float)


class TestComputeWeightHistogram:
    def test_counts_sum_to_numel(self):
        w = torch.randn(64)
        counts, edges = compute_weight_histogram(w, n_bins=16)
        assert abs(sum(counts) - w.numel()) < 1e-3

    def test_correct_bin_count(self):
        w = torch.randn(32)
        counts, edges = compute_weight_histogram(w, n_bins=16)
        assert len(counts) == 16
        assert len(edges) == 17

    def test_all_same_value(self):
        w = torch.full((20,), 3.14)
        counts, edges = compute_weight_histogram(w, n_bins=16)
        assert len(counts) == 16
        assert len(edges) == 16 + 1

    def test_empty_tensor(self):
        counts, edges = compute_weight_histogram(torch.empty(0), n_bins=16)
        assert len(counts) == 16
        assert len(edges) == 17

    def test_returns_python_floats(self):
        w = torch.randn(32)
        counts, edges = compute_weight_histogram(w, n_bins=16)
        for v in counts:
            assert isinstance(v, float)
        for v in edges:
            assert isinstance(v, float)
