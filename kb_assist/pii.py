"""Redact personal and account data before text leaves the assistant.

Applied to user questions before they reach an (optional) LLM provider and to
incident notes before they become a knowledge article draft. Pattern based, so
it reduces risk; it does not guarantee that every identifier is caught.
"""

from __future__ import annotations

import re

_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("NATIONAL_ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    # groups must be separated, so plain digit runs fall through to ACCOUNT
    ("PHONE", re.compile(r"(?<![\w.])\+?\d{1,3}[ .-]\(?\d{2,4}\)?(?:[ .-]\d{3,4}){2,3}\b")),
    ("ACCOUNT", re.compile(r"\b\d{8,17}\b")),
]


def redact(text: str) -> tuple[str, dict[str, int]]:
    """Return (redacted_text, {kind: count})."""
    counts: dict[str, int] = {}
    for kind, pattern in _PATTERNS:
        text, n = pattern.subn(f"[REDACTED_{kind}]", text)
        if n:
            counts[kind] = counts.get(kind, 0) + n
    return text, counts
