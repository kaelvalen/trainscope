"""Alerting backends for TrainScope spike notifications."""

from __future__ import annotations

import json
import logging
import smtplib
import urllib.error
import urllib.request
from email.message import EmailMessage
from typing import Any

logger = logging.getLogger(__name__)


class NullAlerter:
    """No-op alerter used when alerting is disabled or for tests."""

    def alert(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any] | None = None,
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Do nothing."""
        return None


class SlackAlerter:
    """Send spike notifications to a Slack incoming webhook.

    The implementation uses only stdlib so no extra dependencies are required.
    """

    def __init__(self, webhook_url: str, timeout: float = 10.0):
        self._webhook_url = webhook_url
        self._timeout = timeout

    def alert(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any] | None = None,
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        step = spike_info.get("step", "unknown")
        z_score = spike_info.get("z_score", 0.0)
        loss = spike_info.get("loss", 0.0)

        text = (
            f":warning: *TrainScope spike detected* at step {step}\n"
            f"• Loss: `{loss}`\n"
            f"• Z-score: `{z_score}`"
        )
        if global_snap:
            grad_norm = global_snap.get("grad_norm_before_clip")
            if grad_norm is not None:
                text += f"\n• Grad norm: `{grad_norm}`"

        payload = json.dumps({"text": text}).encode("utf-8")
        request = urllib.request.Request(
            self._webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                response.read()
        except urllib.error.URLError:
            logger.exception("Failed to send Slack alert for spike at step %s", step)


class EmailAlerter:
    """Send spike notifications via SMTP using only the standard library."""

    def __init__(
        self,
        to: str | list[str],
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        from_addr: str = "trainscope@localhost",
        subject: str = "TrainScope spike alert",
        use_tls: bool = True,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 10.0,
    ):
        self._to = [to] if isinstance(to, str) else list(to)
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._from_addr = from_addr
        self._subject = subject
        self._use_tls = use_tls
        self._username = username
        self._password = password
        self._timeout = timeout

    def _build_message(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any] | None,
        layer_snaps: dict[str, dict[str, Any]] | None,
    ) -> EmailMessage:
        step = spike_info.get("step", "unknown")
        z_score = spike_info.get("z_score", 0.0)
        loss = spike_info.get("loss", 0.0)

        body_lines = [
            f"TrainScope detected a loss spike at step {step}.",
            "",
            f"Loss: {loss}",
            f"Z-score: {z_score}",
        ]
        if global_snap:
            for key in ("grad_norm_before_clip", "lr", "batch_index"):
                value = global_snap.get(key)
                if value is not None:
                    body_lines.append(f"{key}: {value}")
        if layer_snaps:
            body_lines.append("")
            body_lines.append(f"Layers recorded: {len(layer_snaps)}")

        message = EmailMessage()
        message["From"] = self._from_addr
        message["To"] = ", ".join(self._to)
        message["Subject"] = f"{self._subject} (step {step})"
        message.set_content("\n".join(body_lines))
        return message

    def alert(
        self,
        spike_info: dict[str, Any],
        global_snap: dict[str, Any] | None = None,
        layer_snaps: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        message = self._build_message(spike_info, global_snap, layer_snaps)
        try:
            with smtplib.SMTP(
                self._smtp_host, self._smtp_port, timeout=self._timeout
            ) as server:
                if self._use_tls:
                    server.starttls()
                if self._username is not None and self._password is not None:
                    server.login(self._username, self._password)
                server.send_message(message)
        except Exception:
            logger.exception("Failed to send email alert for spike at step %s", spike_info.get("step"))

