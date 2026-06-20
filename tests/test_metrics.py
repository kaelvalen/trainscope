import math

import pytest
import torch

from trainscope.core.metrics import (
    compute_activation_metrics,
    compute_gradient_metrics,
    compute_weight_histogram,
    compute_weight_metrics,
)


class TestComputeActivationMetrics:
    def test_normal_tensor_finite(self):
        t = torch.randn(16, 32)
        m = compute_activation_metrics(t)
        assert all(math.isfinite(v) for v in m.values())
        assert set(m.keys()) == {
            "act_mean",
            "act_std",
            "act_max_abs",
            "act_kurtosis",
            "act_min",
            "act_max",
            "act_median",
        }

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

    def test_histogram_with_nan_inf(self):
        w = torch.tensor([1.0, 2.0, float("nan"), float("inf")])
        counts, edges = compute_weight_histogram(w, n_bins=4)
        assert len(counts) == 4
        assert len(edges) == 5
        assert sum(counts) == 2.0


class TestNewMetricFields:
    def test_activation_min_max_median(self):
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m = compute_activation_metrics(t)
        assert m["act_min"] == 1.0
        assert m["act_max"] == 4.0
        # torch.median returns the lower median for an even-length tensor.
        assert m["act_median"] == 2.0

    def test_gradient_max_abs_and_mean(self):
        g = torch.tensor([-3.0, 1.0, 2.0])
        m = compute_gradient_metrics(g)
        assert m["grad_max_abs"] == 3.0
        assert m["grad_mean"] == 0.0

    def test_weight_mean_std_min_max(self):
        w = torch.tensor([1.0, 2.0, 3.0, 4.0])
        m = compute_weight_metrics(w)
        assert m["weight_mean"] == 2.5
        assert abs(m["weight_std"] - 1.1180) < 1e-3
        assert m["weight_max_abs"] == 4.0
        assert m["weight_min"] == 1.0


class TestDeviceOffloading:
    def test_cpu_offloading_from_cuda_if_available(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        t = torch.randn(4, 4, device="cuda")
        m = compute_activation_metrics(t, device="cpu")
        assert isinstance(m["act_mean"], float)
        assert m["act_mean"] == pytest.approx(t.float().mean().item())

    def test_same_device_when_requested(self):
        t = torch.randn(4, 4)
        m = compute_weight_metrics(t, device="cpu")
        assert m["weight_mean"] == pytest.approx(t.mean().item())
