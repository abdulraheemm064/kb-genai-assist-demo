"""Draft a knowledge article from a resolved incident (template based).

This is the "incident -> knowledge" step in knowledge-centred service: when an
incident is resolved with a reusable fix, the resolver gets a pre-filled draft
instead of a blank form. The draft:

* is assembled from incident fields with a fixed template (no generation),
* has personal and account data redacted from free text,
* checks the existing knowledge base for a similar article and recommends
  updating it instead of creating a duplicate,
* lists the gaps a reviewer must fill before publishing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, timedelta

from kb_assist.kb import Article
from kb_assist.pii import redact
from kb_assist.retrieval import BM25Index
from kb_assist.text import split_sentences

REQUIRED_INCIDENT_FIELDS = ("number", "short_description", "close_notes")
REUSABLE_RESOLUTIONS = {"Solved (Permanently)", "Solved (Work Around)"}


@dataclass
class Draft:
    article: Article
    markdown: str
    source_incident: str
    similar: list[tuple[str, str, float]] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)
    redactions: dict[str, int] = field(default_factory=dict)
    recommendation: str = "create"  # create | update_existing | not_suitable


def _steps(close_notes: str) -> list[str]:
    return split_sentences(close_notes)


def _title(short_description: str) -> str:
    title = re.sub(r"\s+", " ", short_description).strip().rstrip(".")
    return title[:1].upper() + title[1:]


def draft_from_incident(incident: dict, index: BM25Index, as_of: date,
                        similarity_threshold: float = 12.0, validity_days: int = 365) -> Draft:
    missing = [f for f in REQUIRED_INCIDENT_FIELDS if not incident.get(f)]
    if missing:
        raise ValueError(f"incident is missing required fields: {', '.join(missing)}")

    total_redactions: dict[str, int] = {}

    def clean(text: str) -> str:
        cleaned, counts = redact(text or "")
        for k, v in counts.items():
            total_redactions[k] = total_redactions.get(k, 0) + v
        return cleaned.strip()

    title = _title(clean(incident["short_description"]))
    symptoms = clean(incident.get("description", ""))
    cause = clean(incident.get("cause", ""))
    steps = [clean(s) for s in _steps(incident["close_notes"])]
    workaround = incident.get("resolution_code") == "Solved (Work Around)"

    review_notes: list[str] = []
    if not cause:
        review_notes.append("Cause is empty - add the root cause before publishing.")
    if len(steps) < 2:
        review_notes.append("Resolution has fewer than 2 steps - expand it so another analyst can repeat it.")
    if not incident.get("category"):
        review_notes.append("Category is empty - choose a knowledge category.")
    if not incident.get("service") and not incident.get("cmdb_ci"):
        review_notes.append("No service or CI recorded - add 'Applies to' so the article is found.")
    if workaround:
        review_notes.append("Resolved with a workaround - link the problem record and mark the article "
                            "as a known error.")
    if total_redactions:
        review_notes.append("Personal or account data was redacted from the incident text - check the "
                            "wording still makes sense.")

    lines = [f"# {title}", ""]
    lines += ["## Symptoms", symptoms or "_To be completed by reviewer._", ""]
    env = ", ".join(x for x in (incident.get("service"), incident.get("cmdb_ci")) if x)
    if env:
        lines += ["## Applies to", env, ""]
    lines += ["## Cause", cause or "_To be completed by reviewer._", ""]
    lines += ["## Workaround" if workaround else "## Resolution"]
    lines += [f"{i}. {s}" for i, s in enumerate(steps, 1)] or ["_To be completed by reviewer._"]
    lines += ["", "---", f"_Draft generated from {incident['number']} on {as_of.isoformat()}. "
              "Requires review before publishing._", ""]
    markdown = "\n".join(lines)

    body = markdown.split("\n", 2)[2]  # without the H1 title
    article = Article(
        number=f"DRAFT-{incident['number']}",
        title=title,
        body=body,
        kb_base="IT Service Desk",
        category=incident.get("category", ""),
        owner_group=incident.get("assignment_group", ""),
        author="kb.assist.draft",
        workflow_state="draft",
        valid_to=(as_of + timedelta(days=validity_days)).isoformat(),
        updated_on=as_of.isoformat(),
        tags=[t for t in (incident.get("service", "").lower(), incident.get("category", "").lower()) if t],
    )

    hits = index.search(f"{title} {symptoms}", top_k=3)
    # absolute floor, plus a relative cut so only clear matches are suggested
    similar = [(h.article.number, h.article.title, h.score) for h in hits
               if h.score >= similarity_threshold and h.score >= 0.75 * hits[0].score]
    recommendation = "create"
    if incident.get("resolution_code") not in REUSABLE_RESOLUTIONS:
        recommendation = "not_suitable"
        review_notes.append("Resolution code does not indicate a reusable fix.")
    elif similar:
        recommendation = "update_existing"
        review_notes.insert(0, f"Similar article {similar[0][0]} exists - consider updating it instead "
                               "of publishing a new article.")

    return Draft(article, markdown, incident["number"], similar, review_notes, total_redactions,
                 recommendation)
