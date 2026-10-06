"""BM25 retrieval over knowledge articles, implemented from scratch.

Fields are weighted by repeating their tokens: title x3, tags x2, body x1.
Only published articles are indexed by default, so drafts and retired
content never reach an answer.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

from kb_assist.kb import Article
from kb_assist.text import tokenize

FIELD_WEIGHTS = {"title": 3, "tags": 2, "body": 1}


@dataclass(frozen=True)
class Hit:
    article: Article
    score: float
    matched_terms: tuple[str, ...]


class BM25Index:
    def __init__(self, articles: list[Article], k1: float = 1.5, b: float = 0.75,
                 include_unpublished: bool = False) -> None:
        self.k1, self.b = k1, b
        self.articles = [a for a in articles if include_unpublished or a.searchable]
        self.doc_terms: list[Counter] = [Counter(self._doc_tokens(a)) for a in self.articles]
        self.doc_len = [sum(c.values()) for c in self.doc_terms]
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        df: Counter = Counter()
        for terms in self.doc_terms:
            df.update(terms.keys())
        n = len(self.articles)
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    @staticmethod
    def _doc_tokens(a: Article) -> list[str]:
        return (tokenize(a.title) * FIELD_WEIGHTS["title"]
                + tokenize(" ".join(a.tags)) * FIELD_WEIGHTS["tags"]
                + tokenize(a.body) * FIELD_WEIGHTS["body"])

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        q_terms = list(dict.fromkeys(tokenize(query)))  # unique, keep order
        hits = []
        for idx, terms in enumerate(self.doc_terms):
            score = 0.0
            matched = []
            norm = self.k1 * (1 - self.b + self.b * self.doc_len[idx] / (self.avg_len or 1))
            for t in q_terms:
                tf = terms.get(t, 0)
                if tf:
                    matched.append(t)
                    score += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                hits.append(Hit(self.articles[idx], round(score, 3), tuple(matched)))
        hits.sort(key=lambda h: (-h.score, h.article.number))
        return hits[:top_k]

    def term_idf(self, term: str) -> float:
        return self.idf.get(term, 0.0)
