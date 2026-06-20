import sys
from pathlib import Path

# Ensure the repository root is at the front of sys.path so tests run against
# the local source tree rather than any stale/distro copy on PYTHONPATH.
_ROOT = Path(__file__).resolve().parent.parent
_ROOT_STR = str(_ROOT)
sys.path = [_ROOT_STR] + [p for p in sys.path if p != _ROOT_STR]

# Allow running tests in environments where dev dependencies were installed into
# a local ``.deps`` target directory (e.g. read-only Nix interpreters).
_LOCAL_DEPS = _ROOT / ".deps"
if _LOCAL_DEPS.is_dir():
    sys.path.insert(0, str(_LOCAL_DEPS))

import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402


@pytest.fixture
def simple_model():
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


@pytest.fixture
def simple_optimizer(simple_model):
    return torch.optim.Adam(simple_model.parameters(), lr=1e-3)
