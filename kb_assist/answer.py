"""Compose a grounded answer with citations.

Default mode is *extractive*: the answer is made only of sentences taken from
the retrieved articles, each tagged with a citation like [1]. Nothing is
invented, and when retrieval is weak the assistant abstains and suggests
raising a ticket.

If an LLM provider is enabled, it may rewrite the extractive material into a
smoother answer, but its output is accepted only if it cites the provided
sources and nothing else; otherwise the extractive answer is returned.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from kb_assist.llm import LLMProvider
from kb_assist.pii import redact
from kb_assist.retrieval import BM25Index, Hit
from kb_assist.text import split_sentences, tokenize

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an IT and banking operations knowledge assistant. Answer ONLY from the numbered "
    "sources provided. Cite every statement with its source number in square brackets, e.g. [1]. "
    "If the sources do not answer the question, reply exactly: NO_ANSWER. Never ask for or repeat "
    "passwords, recovery keys or customer account details."
)
ANSWER_SECTIONS = ("resolution", "summary", "body", "symptoms", "cause")


@dataclass
class Source:
    ref: int
    number: str
    title: str
    score: float


@dataclass
class Answer:
    query: str
    text: str
    sources: list[Source] = field(default_factory=list)
    confidence: str = "none"  # high | medium | low | none
    mode: str = "extractive"  # extractive | llm | abstained
    redactions: dict[str, int] = field(default_factory=dict)

    @property
    def answered(self) -> bool:
        return self.mode != "abstained"

    def render(self) -> str:
        lines = [self.text, ""]
        if self.sources:
            lines.append("Sources:")
            lines += [f"  [{s.ref}] {s.number} - {s.title}" for s in self.sources]
        lines.append(f"(confidence: {self.confidence}, mode: {self.mode})")
        if self.redactions:
            kinds = ", ".join(f"{k.lower()} x{v}" for k, v in sorted(self.redactions.items()))
            lines.append(f"(redacted from question before processing: {kinds})")
        return "\n".join(lines)


class KnowledgeAssistant:
    def __init__(self, index: BM25Index, provider: LLMProvider | None = None,
                 min_score: float = 3.0, max_sources: int = 2, max_sentences: int = 5) -> None:
        self.index = index
        self.provider = provider
        self.min_score = min_score
        self.max_sources = max_sources
        self.max_sentences = max_sentences

    def _confidence(self, hits: list[Hit]) -> str:
        top = hits[0].score
        runner_up = hits[1].score if len(hits) > 1 else 0.0
        if top >= 2 * self.min_score and top >= 1.3 * runner_up:
            return "high"
        if top >= 1.5 * self.min_score:
            return "medium"
        return "low"

    def _pick_sentences(self, query_terms: set[str], hits: list[Hit]) -> list[tuple[int, str]]:
        scored = []
        for ref, hit in enumerate(hits, start=1):
            sections = hit.article.sections()
            order = 0
            for name in ANSWER_SECTIONS:
                for sentence in split_sentences(sections.get(name, "")):
                    terms = set(tokenize(sentence))
                    overlap = sum(self.index.term_idf(t) for t in terms & query_terms)
                    in_resolution = 1.0 if name in ("resolution", "summary", "body") else 0.0
                    # Prefer the top article, resolution steps, and query-term overlap.
                    weight = overlap + 2.0 * in_resolution + (1.5 if ref == 1 else 0.0)
                    scored.append((weight, ref, order, sentence))
                    order += 1
        best = sorted(scored, key=lambda s: -s[0])[: self.max_sentences]
        best.sort(key=lambda s: (s[1], s[2]))  # restore reading order within each source
        return [(ref, sentence) for _, ref, _, sentence in best]

    def ask(self, query: str) -> Answer:
        clean_query, redactions = redact(query)
        hits = self.index.search(clean_query, top_k=self.max_sources + 1)
        if not hits or hits[0].score < self.min_score:
            return Answer(clean_query, "I could not find a knowledge article that answers this. "
                          "Please raise a ticket with the service desk so an analyst can help.",
                          [], "none", "abstained", redactions)
        # keep secondary sources only if reasonably close to the best one
        hits = [h for h in hits if h.score >= 0.5 * hits[0].score][: self.max_sources]
        sources = [Source(i, h.article.number, h.article.title, h.score) for i, h in enumerate(hits, 1)]
        picked = self._pick_sentences(set(tokenize(clean_query)), hits)
        text = "\n".join(f"- {sentence} [{ref}]" for ref, sentence in picked)
        answer = Answer(clean_query, text, sources, self._confidence(hits), "extractive", redactions)
        if self.provider is not None:
            self._rewrite_with_llm(answer, hits)
        return answer

    def _rewrite_with_llm(self, answer: Answer, hits: list[Hit]) -> None:
        context = "\n\n".join(f"[{i}] {h.article.number} {h.article.title}\n{h.article.body}"
                              for i, h in enumerate(hits, 1))
        prompt = f"Question: {answer.query}\n\nSources:\n{context}\n\nAnswer with citations."
        try:
            candidate = self.provider.generate(SYSTEM_PROMPT, prompt).strip()
        except Exception as exc:  # provider outages must never break the assistant
            log.warning("LLM provider failed (%s); using extractive answer", type(exc).__name__)
            return
        if is_grounded(candidate, len(answer.sources)):
            answer.text, answer.mode = candidate, "llm"
        else:
            log.warning("LLM answer rejected by citation guardrail; using extractive answer")


def is_grounded(text: str, n_sources: int) -> bool:
    """Accept generated text only if it cites at least one provided source and no others."""
    if not text or text.strip() == "NO_ANSWER":
        return False
    cited = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return bool(cited) and cited <= set(range(1, n_sources + 1))
