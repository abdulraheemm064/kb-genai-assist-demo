"""Command line interface.

    python -m kb_assist ask "VPN fails with error 809"
    python -m kb_assist draft --incident INC0091001
    python -m kb_assist audit --as-of 2026-10-01
    python -m kb_assist demo --out docs/sample-output.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from kb_assist import __version__
from kb_assist.answer import KnowledgeAssistant
from kb_assist.drafting import draft_from_incident
from kb_assist.kb import DEFAULT_KB, load_articles
from kb_assist.llm import LLMConfigError, provider_from_env
from kb_assist.quality import RULES, audit, summarize
from kb_assist.retrieval import BM25Index

log = logging.getLogger("kb_assist")
DEFAULT_INCIDENTS = DEFAULT_KB.parent / "resolved_incidents.json"
DISCLAIMER = ("Representative portfolio project built with synthetic data. "
              "Not derived from any employer or client code.")
DEMO_QUESTIONS = [
    "VPN keeps failing with error 809 when I work from home",
    "card authorization is timing out, what should I do first?",
    "my laptop is asking for a BitLocker recovery key",
    "customer 123456789012 says their online banking account is locked",
    "what is the capital of France?",
]


def _iso(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not an ISO date: {value}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kb_assist", description="Knowledge assistant demo (synthetic KB)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--kb", default=str(DEFAULT_KB), help="knowledge article JSON file")
    sub = p.add_subparsers(dest="command", required=True)

    ask = sub.add_parser("ask", help="answer a question with citations")
    ask.add_argument("question")
    ask.add_argument("--json", action="store_true")

    draft = sub.add_parser("draft", help="draft a KB article from a resolved incident")
    draft.add_argument("--incident", required=True, help="incident number")
    draft.add_argument("--incidents", default=str(DEFAULT_INCIDENTS))
    draft.add_argument("--as-of", type=_iso, default=date.today())
    draft.add_argument("--out", help="write the Markdown draft to this file")

    aud = sub.add_parser("audit", help="knowledge base quality checks")
    aud.add_argument("--as-of", type=_iso, default=date.today())
    aud.add_argument("--json", action="store_true")

    demo = sub.add_parser("demo", help="regenerate the sample output document")
    demo.add_argument("--as-of", type=_iso, default=date(2026, 10, 1))
    demo.add_argument("--out", default="docs/sample-output.md")
    return p


def _assistant(articles) -> KnowledgeAssistant:
    provider = provider_from_env()
    if provider:
        log.info("LLM provider enabled: %r", provider)
    return KnowledgeAssistant(BM25Index(articles), provider=provider)


def cmd_ask(args, articles) -> int:
    answer = _assistant(articles).ask(args.question)
    if args.json:
        print(json.dumps(asdict(answer), indent=2))
    else:
        print(answer.render())
    return 0 if answer.answered else 3


def cmd_draft(args, articles) -> int:
    incidents = {i["number"]: i for i in json.loads(Path(args.incidents).read_text(encoding="utf-8"))}
    if args.incident not in incidents:
        log.error("Incident %s not found in %s", args.incident, args.incidents)
        return 1
    draft = draft_from_incident(incidents[args.incident], BM25Index(articles), args.as_of)
    print(draft.markdown)
    print(f"Recommendation: {draft.recommendation}")
    for note in draft.review_notes:
        print(f"  - {note}")
    if args.out:
        Path(args.out).write_text(draft.markdown, encoding="utf-8")
    return 0


def _audit_markdown(articles, issues, as_of: date) -> list[str]:
    s = summarize(articles, issues)
    out = [f"Audited {s['articles_audited']} active articles (of {s['articles_total']}) as of "
           f"{as_of.isoformat()}: **{s['articles_with_issues']} with issues, "
           f"{s['healthy_pct']}% healthy**.", "",
           "| Rule | Severity | Article | Title | Detail |", "|---|---|---|---|---|"]
    out += [f"| {i.rule_id} {RULES[i.rule_id][1]} | {i.severity} | {i.number} | {i.title} | {i.detail} |"
            for i in issues]
    return out


def cmd_audit(args, articles) -> int:
    issues = audit(articles, args.as_of)
    if args.json:
        print(json.dumps({"summary": summarize(articles, issues),
                          "issues": [i.to_dict() for i in issues]}, indent=2))
    else:
        print("\n".join(_audit_markdown(articles, issues, args.as_of)))
    return 0


def cmd_demo(args, articles) -> int:
    assistant = KnowledgeAssistant(BM25Index(articles))  # demo output is always offline/extractive
    out = ["# Sample output", "", f"> {DISCLAIMER}", "",
           "Generated by `python -m kb_assist demo`. Extractive mode, no LLM provider.", "",
           "## 1. Questions answered from the knowledge base", ""]
    for q in DEMO_QUESTIONS:
        out += [f"### Q: {q}", "", "```text", assistant.ask(q).render(), "```", ""]
    out += ["## 2. Knowledge drafts from resolved incidents", ""]
    incidents = json.loads(DEFAULT_INCIDENTS.read_text(encoding="utf-8"))
    for inc in incidents:
        d = draft_from_incident(inc, BM25Index(articles), args.as_of)
        out += [f"### From {inc['number']} - recommendation: `{d.recommendation}`", "",
                "```markdown", d.markdown.rstrip(), "```", "", "Review notes:", ""]
        out += [f"- {n}" for n in d.review_notes] + [""]
    out += ["## 3. Knowledge base quality audit", ""]
    out += _audit_markdown(articles, audit(articles, args.as_of), args.as_of) + [""]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    try:
        articles = load_articles(args.kb)
        handler = {"ask": cmd_ask, "draft": cmd_draft, "audit": cmd_audit, "demo": cmd_demo}[args.command]
        return handler(args, articles)
    except (LLMConfigError, ValueError, OSError) as exc:
        log.error("%s", exc)
        return 1
