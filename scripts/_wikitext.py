"""Shared wikitext-2 dataset resolution for the verification scripts.

The verification experiments all train on the wikitext-2 raw dataset cached
by the Hugging Face ``datasets`` library. Instead of hardcoding any single
developer's absolute cache path, this module locates the cached Arrow file
from the standard HF cache layout (``~/.cache/huggingface/datasets/...``) and
returns a clear error when it is absent.

Callers that need a specific location can always pass ``--data``; resolution
is only the *default* fallback.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path


def find_wikitext_arrow() -> str:
    """Return the path to the cached wikitext-2 raw train Arrow file.

    Searches the standard Hugging Face datasets cache under the user's home
    directory. Raises ``FileNotFoundError`` with an actionable message when
    the dataset has not been downloaded.
    """
    patterns = [
        "~/.cache/huggingface/datasets/**/wikitext-train.arrow",
        "~/.cache/huggingface/datasets/**/wikitext-2-raw-v1/**/wikitext-train.arrow",
    ]
    for pattern in patterns:
        matches = sorted(glob.glob(os.path.expanduser(pattern), recursive=True))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "wikitext-2 raw dataset not found in the Hugging Face cache "
        f"({Path('~/.cache/huggingface/datasets').expanduser()}). "
        "Download it first, e.g.:\n"
        "  from datasets import load_dataset\n"
        "  load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1')\n"
        "or pass --data /path/to/wikitext-train.arrow"
    )


__all__ = ["find_wikitext_arrow"]
