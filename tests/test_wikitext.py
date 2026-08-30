"""Tests for the verification scripts' shared wikitext-2 resolution.

The verification experiments reference a cached wikitext-2 Arrow file; the
shared helper must find it in the standard HF cache layout, and raise an
actionable error (not a developer's hardcoded path) when it is absent.
"""

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def wikitext_module(tmp_path):
    sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    module = importlib.import_module("_wikitext")
    yield module
    sys.path.remove(str(Path(__file__).parent.parent / "scripts"))


def test_finds_cached_wikitext_in_hf_layout(tmp_path, monkeypatch, wikitext_module):
    """The helper locates wikitext-train.arrow under a standard HF-style
    cache tree without hardcoding any user's home directory."""
    dataset_dir = tmp_path / ".cache" / "huggingface" / "datasets"
    nested = dataset_dir / "Salesforce___wikitext" / "wikitext-2-raw-v1" / "0.0.0"
    nested.mkdir(parents=True)
    arrow = nested / "wikitext-train.arrow"
    arrow.write_bytes(b"fake-arrow")

    monkeypatch.setenv("HOME", str(tmp_path))
    assert wikitext_module.find_wikitext_arrow() == str(arrow)


def test_missing_dataset_raises_actionable_error(monkeypatch, wikitext_module):
    """When the dataset has not been downloaded, the error must explain how
    to obtain it — never reveal a developer's machine path."""
    monkeypatch.setenv("HOME", str(Path("/nonexistent_home")))
    with pytest.raises(FileNotFoundError) as excinfo:
        wikitext_module.find_wikitext_arrow()
    message = str(excinfo.value)
    assert "wikitext-2 raw dataset not found" in message
    assert "load_dataset" in message
    assert "--data" in message
    # No developer-specific path leaks into the message.
    assert "/home/kael" not in message
