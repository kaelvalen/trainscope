"""End-to-end integration test for TrainScope.

Runs a tiny frozen training loop with an injected loss spike, verifies that
all expected artifacts are created, and checks that the FastAPI server can read
the run directory.
"""

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastapi.testclient import TestClient

from trainscope import StopTraining, TrainScope
from trainscope.core.config import TrainScopeConfig
from trainscope.ui.server import create_app

SPIKE_STEP = 45
SPIKE_MULTIPLIER = 100.0
N_STEPS = 60


def _run_training(run_path: Path) -> dict | None:
    """Run a small deterministic loop and return the detected spike info."""
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 1),
    )
    # Use a zero learning rate so the baseline loss stays nearly constant,
    # making the injected spike easy to detect.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

    config = TrainScopeConfig(
        run_dir=str(run_path),
        run_name="integration_run",
        detector={"name": "z_score", "threshold": 3.5},
        track_memory=True,
        checkpoint_on_spike=True,
        rng_every_n_steps=10,
    )

    scope = TrainScope(model, optimizer, config).attach()

    # Random inputs and a zero learning rate produce a stable but non-constant
    # loss baseline, so the injected spike has a finite z-score.
    y = torch.randn(4, 1)

    detected_spike = None
    for step in range(N_STEPS):
        optimizer.zero_grad()
        x = torch.randn(4, 8)
        loss = F.mse_loss(model(x), y)

        if step == SPIKE_STEP:
            loss = loss * SPIKE_MULTIPLIER

        loss.backward()
        spike = scope.step(loss.item(), batch_index=step)
        optimizer.step()

        if spike is not None:
            detected_spike = spike

    scope.writer.flush()
    scope.writer.close()
    scope.detach()

    return detected_spike


def test_integration_detects_spike_and_creates_artifacts(tmp_path: Path) -> None:
    spike = _run_training(tmp_path)

    assert spike is not None
    assert spike["step"] == SPIKE_STEP
    assert math.isfinite(spike["spike_score"])
    assert abs(spike["spike_score"]) > 3.5

    run_dir = tmp_path / "integration_run"

    # Core artifacts.
    assert (run_dir / "meta.json").exists()
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "global.arrow").exists()
    assert any((run_dir / "layers").glob("*.arrow"))

    # Spike artifacts.
    assert (run_dir / "spikes" / f"spike_step_{SPIKE_STEP}.arrow").exists()
    assert (run_dir / "spikes" / f"spike_step_{SPIKE_STEP}_layers").is_dir()
    assert any((run_dir / "spikes" / f"spike_step_{SPIKE_STEP}_layers").glob("*.arrow"))

    # Optional checkpoint artifact.
    assert (run_dir / "checkpoints" / f"{SPIKE_STEP}.pt").exists()

    # Periodic RNG state artifact.
    assert (run_dir / "rng_states" / "step_10.pkl").exists()


def test_integration_spike_invokes_alerter_via_alert_method(tmp_path: Path) -> None:
    """End-to-end: TrainScope.step() must call alerter.alert(), the only
    method every built-in alerter (NullAlerter/SlackAlerter/EmailAlerter)
    actually implements. A regression here (e.g. calling a nonexistent
    ``.notify()``) is swallowed by scope.py's broad except-and-log, so it
    must be caught by going through step() rather than calling alert()
    directly."""
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    config = TrainScopeConfig(run_dir=str(tmp_path), run_name="alert_run")

    scope = TrainScope(model, optimizer, config).attach()

    calls = []

    class RecordingAlerter:
        def alert(self, spike_info, global_snap=None, layer_snaps=None):
            calls.append(spike_info)

    scope._alerters = [RecordingAlerter()]

    y = torch.randn(4, 1)
    for step in range(50):
        optimizer.zero_grad()
        x = torch.randn(4, 8)
        loss = F.mse_loss(model(x), y)
        if step == 40:
            loss = loss * 100.0
        loss.backward()
        scope.step(loss.item(), batch_index=step)
        optimizer.step()

    scope.detach()

    assert len(calls) >= 1
    assert any(c["step"] == 40 for c in calls)


def test_integration_server_reads_run_directory(tmp_path: Path) -> None:
    _run_training(tmp_path)
    run_dir = tmp_path / "integration_run"

    app = create_app(str(run_dir))
    client = TestClient(app)

    health_response = client.get("/api/health")
    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"

    manifest_response = client.get("/api/manifest")
    assert manifest_response.status_code == 200
    manifest = manifest_response.json()
    assert manifest["last_step"] == N_STEPS - 1
    assert manifest["n_global_rows"] == N_STEPS

    meta_response = client.get("/api/meta")
    assert meta_response.status_code == 200
    meta = meta_response.json()
    assert meta["model_name"] == "Sequential"
    assert meta["trainscope_config"]["run_name"] == "integration_run"

    global_response = client.get("/api/global")
    assert global_response.status_code == 200
    global_rows = global_response.json()
    assert len(global_rows) == N_STEPS
    assert any(row["is_spike"] for row in global_rows)

    layers_response = client.get("/api/layers")
    assert layers_response.status_code == 200
    layer_names = layers_response.json()
    assert len(layer_names) > 0
    assert all(isinstance(name, str) for name in layer_names)

    ranked_response = client.get("/api/layers/ranked")
    assert ranked_response.status_code == 200
    ranked = ranked_response.json()
    assert isinstance(ranked, list)
    assert len(ranked) <= len(layer_names)

    spikes_response = client.get("/api/spikes")
    assert spikes_response.status_code == 200
    spikes = spikes_response.json()
    assert any(record["step"] == SPIKE_STEP for record in spikes)

    # Verify the server can read an individual layer.
    first_layer = layer_names[0]
    layer_response = client.get(f"/api/layers/{first_layer}")
    assert layer_response.status_code == 200
    layer_rows = layer_response.json()
    assert len(layer_rows) == N_STEPS

    # Diff endpoint should produce KL divergences for histogram pairs.
    # Histograms are written every 50 steps and on spike steps, so steps 0 and
    # the spike step both have weight histograms for every layer.
    diff_response = client.get(f"/api/diff?step_a=0&step_b={SPIKE_STEP}")
    assert diff_response.status_code == 200
    diffs = diff_response.json()
    assert isinstance(diffs, list)
    assert any(item["layer"] == first_layer for item in diffs)

    # Spike window endpoints.
    spike_layers_response = client.get(f"/api/spikes/{SPIKE_STEP}/layers")
    assert spike_layers_response.status_code == 200
    assert first_layer in spike_layers_response.json()

    spike_layer_response = client.get(f"/api/spikes/{SPIKE_STEP}/layers/{first_layer}")
    assert spike_layer_response.status_code == 200
    assert isinstance(spike_layer_response.json(), list)


def test_integration_stop_training_carries_spike_score(tmp_path: Path) -> None:
    """StopTraining exposes the active detector's score as spike_score."""
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    config = TrainScopeConfig(
        run_dir=str(tmp_path),
        run_name="stop_run",
        stop_on_spike=True,
        detector={"name": "z_score", "threshold": 3.5},
    )

    scope = TrainScope(model, optimizer, config).attach()

    y = torch.randn(4, 1)
    raised = None
    try:
        for step in range(50):
            optimizer.zero_grad()
            x = torch.randn(4, 8)
            loss = F.mse_loss(model(x), y)
            if step == 40:
                loss = loss * 100.0
            loss.backward()
            scope.step(loss.item(), batch_index=step)
            optimizer.step()
    except StopTraining as exc:
        raised = exc

    scope.detach()

    assert raised is not None
    assert raised.spike_score > 3.5


def test_stop_training_z_score_alias_deprecated() -> None:
    """StopTraining.z_score still works but warns it is a legacy alias."""
    exc = StopTraining(step=40, spike_score=7.2)
    assert exc.spike_score == 7.2

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert exc.z_score == 7.2
        assert any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_spike_info_contains_spike_score_and_legacy_alias(tmp_path: Path) -> None:
    """spike_info carries both spike_score (canonical) and z_score (legacy)."""
    torch.manual_seed(0)

    model = nn.Sequential(nn.Linear(8, 4), nn.ReLU(), nn.Linear(4, 1))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    config = TrainScopeConfig(
        run_dir=str(tmp_path),
        run_name="spike_info_run",
        detector={"name": "z_score", "threshold": 3.5},
    )

    scope = TrainScope(model, optimizer, config).attach()

    captured = []

    class RecordingCallback:
        def on_step(self, global_snap=None, layer_snaps=None):
            pass

        def on_spike(self, spike_info, global_snap=None, layer_snaps=None):
            captured.append(spike_info)

    scope._callbacks = [RecordingCallback()]

    y = torch.randn(4, 1)
    for step in range(50):
        optimizer.zero_grad()
        x = torch.randn(4, 8)
        loss = F.mse_loss(model(x), y)
        if step == 40:
            loss = loss * 100.0
        loss.backward()
        scope.step(loss.item(), batch_index=step)
        optimizer.step()

    scope.detach()

    assert len(captured) >= 1
    spike = captured[0]
    assert "spike_score" in spike
    assert "z_score" in spike
    assert spike["spike_score"] == spike["z_score"]
