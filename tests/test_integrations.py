"""Tests for TrainScope integrations and alerting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from trainscope.integrations import build_alerters, build_callbacks
from trainscope.integrations.alerts import EmailAlerter, NullAlerter, SlackAlerter


class FakeWandbRun:
    def __init__(self):
        self.logs: list[tuple[dict, int | None]] = []

    def log(self, metrics, step=None):
        self.logs.append((metrics, step))


def _mock_wandb_module(fake_run: FakeWandbRun) -> MagicMock:
    module = MagicMock()
    module.init.return_value = fake_run
    return module


class TestBuildCallbacks:
    def test_empty_config_returns_empty_list(self):
        assert build_callbacks(None) == []
        assert build_callbacks({}) == []

    def test_unknown_integration_is_skipped(self):
        callbacks = build_callbacks({"unknown": {"foo": "bar"}})
        assert callbacks == []

    def test_wandb_callback(self):
        fake_run = FakeWandbRun()
        module = _mock_wandb_module(fake_run)
        with patch.dict("sys.modules", {"wandb": module}):
            callbacks = build_callbacks({"wandb": {"project": "test"}})

        assert len(callbacks) == 1
        callback = callbacks[0]
        callback.on_step(
            {"step": 5, "loss": 1.2, "grad_norm_before_clip": 0.5, "lr": 0.01},
            {"layer1": {"grad_l2_norm": 0.1, "act_kurtosis": 2.0}},
        )
        assert len(fake_run.logs) == 1
        metrics, step = fake_run.logs[0]
        assert step == 5
        assert metrics["train/loss"] == 1.2

    def test_tensorboard_callback(self):
        mock_writer = MagicMock()
        mock_tb_module = MagicMock()
        mock_tb_module.SummaryWriter = MagicMock(return_value=mock_writer)
        with patch.dict("sys.modules", {"torch.utils.tensorboard": mock_tb_module}):
            callbacks = build_callbacks({"tensorboard": {"log_dir": "/tmp/tb"}})

        assert len(callbacks) == 1
        callback = callbacks[0]
        callback.on_step(
            {"step": 3, "loss": 2.0, "grad_norm_before_clip": 1.0, "lr": 0.001},
            None,
        )
        mock_writer.add_scalar.assert_any_call("train/loss", 2.0, 3)

    def test_mlflow_callback(self):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            callbacks = build_callbacks({"mlflow": {"experiment_name": "exp"}})

        assert len(callbacks) == 1
        callback = callbacks[0]
        callback.on_step(
            {"step": 7, "loss": 0.8, "grad_norm_before_clip": 0.2, "lr": 0.01},
            {"l1": {"grad_l2_norm": 0.3}},
        )
        mock_mlflow.log_metrics.assert_called()
        args, kwargs = mock_mlflow.log_metrics.call_args
        assert "train_loss" in args[0]
        assert kwargs.get("step") == 7


class TestBuildAlerters:
    def test_empty_config_returns_empty_list(self):
        assert build_alerters(None) == []
        assert build_alerters([]) == []

    def test_null_alerter(self):
        alerters = build_alerters([{"type": "null"}])
        assert len(alerters) == 1
        assert isinstance(alerters[0], NullAlerter)
        alerters[0].alert({"step": 1})

    def test_slack_alerter(self):
        alerter = SlackAlerter("https://hooks.slack.com/test")
        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_response = MagicMock()
            mock_urlopen.return_value.__enter__.return_value = mock_response
            alerter.alert({"step": 10, "z_score": 4.5, "loss": 5.0})

        mock_urlopen.assert_called_once()
        request = mock_urlopen.call_args[0][0]
        assert request.full_url == "https://hooks.slack.com/test"
        body = request.data.decode("utf-8")
        assert "spike" in body.lower()
        assert "step 10" in body

    def test_email_alerter(self):
        alerter = EmailAlerter(
            to="ops@example.com",
            smtp_host="smtp.example.com",
            smtp_port=587,
            use_tls=True,
            username="user",
            password="pass",
        )
        with patch("smtplib.SMTP") as mock_smtp:
            instance = mock_smtp.return_value.__enter__.return_value
            alerter.alert({"step": 20, "z_score": 3.5, "loss": 2.0})

        mock_smtp.assert_called_once_with("smtp.example.com", 587, timeout=10.0)
        instance.starttls.assert_called_once()
        instance.login.assert_called_once_with("user", "pass")
        instance.send_message.assert_called_once()


class TestCallbackInterfaces:
    """Verify callbacks expose the required on_step / on_spike interface."""

    def test_wandb_on_spike(self):
        fake_run = FakeWandbRun()
        module = _mock_wandb_module(fake_run)
        with patch.dict("sys.modules", {"wandb": module}):
            callbacks = build_callbacks({"wandb": {"project": "test"}})

        callback = callbacks[0]
        callback.on_spike(
            {"step": 5, "z_score": 4.0, "loss": 1.0},
            {"step": 5, "loss": 1.0},
            None,
        )
        assert any("spike" in str(log[0]) for log in fake_run.logs)

    def test_tensorboard_on_spike(self):
        mock_writer = MagicMock()
        mock_tb_module = MagicMock()
        mock_tb_module.SummaryWriter = MagicMock(return_value=mock_writer)
        with patch.dict("sys.modules", {"torch.utils.tensorboard": mock_tb_module}):
            callbacks = build_callbacks({"tensorboard": {}})

        callbacks[0].on_spike(
            {"step": 2, "z_score": 3.0, "loss": 0.5},
            {"step": 2},
            None,
        )
        mock_writer.add_scalar.assert_any_call("spike/z_score", 3.0, 2)

    def test_mlflow_on_spike(self):
        mock_mlflow = MagicMock()
        with patch.dict("sys.modules", {"mlflow": mock_mlflow}):
            callbacks = build_callbacks({"mlflow": {}})

        callbacks[0].on_spike(
            {"step": 9, "z_score": 5.0, "loss": 0.1},
            {"step": 9},
            None,
        )
        mock_mlflow.log_metrics.assert_called()
