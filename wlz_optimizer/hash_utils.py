"""Stable hashing helpers."""

from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def short_hash(text: str, length: int = 10) -> str:
    return sha256_text(text)[:length]
