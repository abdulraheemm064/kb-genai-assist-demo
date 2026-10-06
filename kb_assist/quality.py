"""Knowledge base quality checks (content governance).

| Rule  | Check |
|-------|-------|
| KQ001 | Expired: valid_to is in the past but the article is still published |
| KQ002 | Review overdue: not updated for more than `review_days` |
| KQ003 | Missing metadata: category, owner group, knowledge base or tags |
| KQ004 | Thin content: body shorter than `min_body_chars` |
| KQ005 | No resolution / summary section |
| KQ006 | Near-duplicate of another article (TF-IDF cosine >= threshold) |
| KQ007 | Low helpfulness: < 40% helpful with at least 10 votes |
| KQ008 | References a retired or unknown article |
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import date

from kb_assist.kb import Article
from kb_assist.text import tokenize

RULES = {
    "KQ001": ("high", "Expired but still published"),
    "KQ002": ("medium", "Review overdue"),
    "KQ003": ("medium", "Missing metadata"),
    "KQ004": ("medium", "Thin content"),
    "KQ005": ("medium", "No resolution section"),
    "KQ006": ("medium", "Near-duplicate article"),
    "KQ007": ("low", "Low helpfulness rating"),
    "KQ008": ("high", "References retired or unknown article"),
}


@dataclass(frozen=True)
class Issue:
    rule_id: str
    severity: str
    number: str
    title: str
    detail: str

    def to_dict(self) -> dict:
        return asdict(self)


def _issue(rule: str, a: Article, detail: str) -> Issue:
    severity, _ = RULES[rule]
    return Issue(rule, severity, a.number, a.title, detail)


def _tfidf(articles: list[Article]) -> list[dict[str, float]]:
    docs = [Counter(tokenize(a.title + " " + a.body)) for a in articles]
    df: Counter = Counter()
    for d in docs:
        df.update(d.keys())
    n = len(docs)
    vectors = []
    for d in docs:
        vec = {t: tf * math.log((1 + n) / (1 + df[t])) for t, tf in d.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def audit(articles: list[Article], as_of: date, review_days: int = 365, min_body_chars: int = 300,
          duplicate_threshold: float = 0.6) -> list[Issue]:
    issues: list[Issue] = []
    states = {a.number: a.workflow_state for a in articles}
    active = [a for a in articles if a.workflow_state != "retired"]

    for a in active:
        if a.workflow_state == "published" and a.valid_to and date.fromisoformat(a.valid_to) < as_of:
            issues.append(_issue("KQ001", a, f"valid_to {a.valid_to}"))
        if a.updated_on:
            age = (as_of - date.fromisoformat(a.updated_on)).days
            if age > review_days:
                issues.append(_issue("KQ002", a, f"last updated {age} days ago"))
        missing = [f for f in ("category", "owner_group", "kb_base") if not getattr(a, f)]
        if not a.tags:
            missing.append("tags")
        if missing:
            issues.append(_issue("KQ003", a, "missing " + ", ".join(missing)))
        if len(a.body) < min_body_chars:
            issues.append(_issue("KQ004", a, f"body is {len(a.body)} characters"))
        sections = a.sections()
        if not any(k in sections for k in ("resolution", "summary", "workaround")):
            issues.append(_issue("KQ005", a, "no Resolution/Summary/Workaround heading"))
        votes = a.helpful_yes + a.helpful_no
        if votes >= 10 and a.helpful_yes / votes < 0.4:
            issues.append(_issue("KQ007", a, f"{a.helpful_yes}/{votes} found it helpful"))
        for ref in sorted(set(re.findall(r"\bKB\d{7}\b", a.body)) - {a.number}):
            if states.get(ref) in (None, "retired"):
                issues.append(_issue("KQ008", a, f"links to {ref} ({states.get(ref, 'unknown')})"))

    vectors = _tfidf(active)
    for i in range(len(active)):
        for j in range(i + 1, len(active)):
            sim = cosine(vectors[i], vectors[j])
            if sim >= duplicate_threshold:
                a, b = active[i], active[j]
                # report on the weaker article (fewer helpful votes) as the merge candidate
                weaker, stronger = (a, b) if a.helpful_yes <= b.helpful_yes else (b, a)
                issues.append(_issue("KQ006", weaker,
                                     f"{sim:.2f} similar to {stronger.number}; merge into it"))
    return sorted(issues, key=lambda x: (x.rule_id, x.number))


def summarize(articles: list[Article], issues: list[Issue]) -> dict:
    active = [a for a in articles if a.workflow_state != "retired"]
    flagged = {i.number for i in issues}
    return {
        "articles_total": len(articles),
        "articles_audited": len(active),
        "articles_with_issues": len(flagged),
        "healthy_pct": round(100 * (1 - len(flagged) / len(active)), 1) if active else 100.0,
        "issues_by_rule": dict(Counter(i.rule_id for i in issues)),
    }
