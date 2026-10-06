import json
from collections import Counter
from datetime import date

import pytest

from kb_assist.cli import main
from kb_assist.drafting import draft_from_incident
from kb_assist.kb import DEFAULT_KB, Article, load_articles
from kb_assist.pii import redact
from kb_assist.quality import audit, cosine, summarize
from kb_assist.retrieval import BM25Index

AS_OF = date(2026, 10, 1)
INCIDENTS = {i["number"]: i for i in json.loads((DEFAULT_KB.parent / "resolved_incidents.json").read_text())}


@pytest.fixture(scope="module")
def articles():
    return load_articles()


@pytest.fixture(scope="module")
def index(articles):
    return BM25Index(articles)


# ---------- PII ----------

@pytest.mark.parametrize("text,kind", [
    ("mail a.b@contoso-bank.example now", "EMAIL"),
    ("card 4111 1111 1111 1111", "CARD"),
    ("id 123-45-6789", "NATIONAL_ID"),
    ("call +1 555 0100 1234", "PHONE"),
    ("account 123456789012", "ACCOUNT"),
])
def test_redaction_kinds(text, kind):
    cleaned, counts = redact(text)
    assert counts == {kind: 1}
    assert f"[REDACTED_{kind}]" in cleaned


def test_redaction_leaves_operational_text_alone():
    text = "Restart at 08:55, driver 5.2, branch 0412, see KB0010007 and INC0091001, port 443"
    assert redact(text) == (text, {})


# ---------- drafting ----------

def test_draft_structure_and_redaction(index):
    d = draft_from_incident(INCIDENTS["INC0091001"], index, AS_OF)
    md = d.markdown
    headings = ("# Branch teller workstations", "## Symptoms", "## Applies to", "## Cause", "## Resolution")
    for heading in headings:
        assert heading in md
    assert "1. Rolled back the generic driver" in md
    assert "teller.lead@" not in md and "123456789012" not in md and "555 0100" not in md
    assert d.redactions == {"EMAIL": 1, "PHONE": 1, "ACCOUNT": 1}
    assert d.article.workflow_state == "draft"
    assert d.article.valid_to == "2027-10-01"


def test_draft_recommends_updating_similar_article(index):
    d = draft_from_incident(INCIDENTS["INC0091001"], index, AS_OF)
    assert d.recommendation == "update_existing"
    assert d.similar[0][0] == "KB0010007"


def test_workaround_draft(index):
    d = draft_from_incident(INCIDENTS["INC0091002"], index, AS_OF)
    assert "## Workaround" in d.markdown
    assert any("known error" in n for n in d.review_notes)


def test_thin_incident_gets_review_notes(index):
    d = draft_from_incident(INCIDENTS["INC0091003"], index, AS_OF)
    notes = " ".join(d.review_notes)
    for expected in ("Cause is empty", "fewer than 2 steps", "Category is empty", "No service or CI"):
        assert expected in notes
    assert "_To be completed by reviewer._" in d.markdown


def test_new_topic_is_created_and_non_reusable_is_flagged(index):
    incident = {"number": "INC1", "short_description": "Quantum widget overheats",
                "close_notes": "1. Replaced fan.\n2. Verified temperature.",
                "resolution_code": "Solved (Permanently)", "category": "Hardware", "cause": "Fan failed"}
    assert draft_from_incident(incident, index, AS_OF).recommendation == "create"
    incident["resolution_code"] = "Closed/Resolved by Caller"
    assert draft_from_incident(incident, index, AS_OF).recommendation == "not_suitable"


def test_draft_requires_fields(index):
    with pytest.raises(ValueError, match="close_notes"):
        draft_from_incident({"number": "INC2", "short_description": "x"}, index, AS_OF)


# ---------- quality ----------

def test_audit_finds_seeded_issues(articles):
    issues = audit(articles, AS_OF)
    by_rule = Counter(i.rule_id for i in issues)
    assert by_rule == {"KQ001": 2, "KQ002": 3, "KQ003": 2, "KQ004": 1, "KQ006": 1, "KQ007": 2, "KQ008": 1}
    dup = next(i for i in issues if i.rule_id == "KQ006")
    assert dup.number == "KB0010003" and "KB0010002" in dup.detail
    assert all(i.number != "KB0010024" for i in issues)  # retired articles are not audited
    summary = summarize(articles, issues)
    assert summary["articles_audited"] == 24 and summary["articles_with_issues"] == 5


def test_audit_thresholds_are_configurable(articles):
    assert not [i for i in audit(articles, AS_OF, review_days=1000) if i.rule_id == "KQ002"]
    assert len([i for i in audit(articles, AS_OF, duplicate_threshold=0.99) if i.rule_id == "KQ006"]) == 0


def test_missing_resolution_section():
    a = Article("KB9", "t", "## Symptoms\n" + "x " * 200, category="c", owner_group="g", kb_base="b",
                tags=["t"], workflow_state="published", updated_on="2026-09-01")
    assert [i.rule_id for i in audit([a], AS_OF)] == ["KQ005"]


def test_cosine_bounds():
    v = {"a": 0.6, "b": 0.8}
    assert cosine(v, v) == pytest.approx(1.0)
    assert cosine(v, {"c": 1.0}) == 0.0


# ---------- CLI ----------

def test_cli_ask(capsys):
    assert main(["ask", "teller receipt printer not printing"]) == 0
    assert "KB0010007" in capsys.readouterr().out
    assert main(["ask", "what is the capital of France?"]) == 3


def test_cli_ask_json(capsys):
    assert main(["ask", "renew tls certificate", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["sources"][0]["number"] == "KB0010011"


def test_cli_draft_audit_demo(tmp_path, capsys):
    out = tmp_path / "draft.md"
    assert main(["draft", "--incident", "INC0091002", "--as-of", "2026-10-01", "--out", str(out)]) == 0
    assert out.read_text().startswith("# Payments Gateway API timeouts")
    assert main(["draft", "--incident", "INC404"]) == 1
    assert main(["audit", "--as-of", "2026-10-01", "--json"]) == 0
    assert '"KQ006": 1' in capsys.readouterr().out
    demo = tmp_path / "sample.md"
    assert main(["demo", "--out", str(demo)]) == 0
    text = demo.read_text()
    assert "synthetic data" in text and "## 3. Knowledge base quality audit" in text


def test_cli_misconfigured_llm_is_reported(monkeypatch):
    monkeypatch.setenv("KB_ASSIST_LLM_ENABLED", "true")
    monkeypatch.setenv("KB_ASSIST_LLM_ENDPOINT", "http://insecure.example")
    assert main(["ask", "vpn error 809"]) == 1
