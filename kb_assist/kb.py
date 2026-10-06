"""Knowledge article model and loader."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_KB = Path(__file__).resolve().parent.parent / "data" / "kb_articles.json"
SEARCHABLE_STATES = {"published"}


@dataclass
class Article:
    number: str
    title: str
    body: str
    kb_base: str = ""
    category: str = ""
    owner_group: str = ""
    author: str = ""
    workflow_state: str = "draft"
    valid_to: str = ""
    updated_on: str = ""
    tags: list[str] = field(default_factory=list)
    helpful_yes: int = 0
    helpful_no: int = 0

    @property
    def searchable(self) -> bool:
        return self.workflow_state in SEARCHABLE_STATES

    def sections(self) -> dict[str, str]:
        """Split a Markdown body on '## Heading' lines -> {heading_lower: text}."""
        parts: dict[str, str] = {}
        current = "body"
        for line in self.body.splitlines():
            match = re.match(r"^##\s+(.+?)\s*$", line)
            if match:
                current = match.group(1).strip().lower()
                parts.setdefault(current, "")
            else:
                parts[current] = (parts.get(current, "") + "\n" + line).strip()
        return {k: v for k, v in parts.items() if v}


def load_articles(path: str | Path = DEFAULT_KB) -> list[Article]:
    with Path(path).open(encoding="utf-8") as fh:
        rows = json.load(fh)
    articles = [Article(**row) for row in rows]
    numbers = [a.number for a in articles]
    if len(numbers) != len(set(numbers)):
        raise ValueError("duplicate article numbers in knowledge base")
    return articles
