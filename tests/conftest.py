import pytest
import torch
import torch.nn as nn


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
