"""Tokenisation, light stemming and sentence splitting (no external NLP deps)."""

from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an and are as at be been but by can cannot do does for from has have how i if in into is it
its my of on or our so that the their then there these this to was we were what when where which
who why will with you your after before not no yes please get got any all also than too very
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Very small suffix stripper: good enough to match 'printing'/'printer'/'prints'."""
    for suffix in ("ations", "ation", "ings", "ing", "ers", "er", "ies", "es", "ed", "s"):
        if len(token) > len(suffix) + 3 and token.endswith(suffix):
            return token[: -len(suffix)] + ("y" if suffix == "ies" else "")
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split into sentences / numbered steps, dropping list markers."""
    out = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", line).strip()
        if not line:
            continue
        for piece in re.split(r"(?<=[.!?])\s+(?=[A-Z])", line):
            piece = piece.strip()
            if len(piece) > 3:
                out.append(piece)
    return out
