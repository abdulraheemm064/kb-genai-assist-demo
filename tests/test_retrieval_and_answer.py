import pytest

from kb_assist.answer import KnowledgeAssistant, is_grounded
from kb_assist.kb import Article, load_articles
from kb_assist.retrieval import BM25Index
from kb_assist.text import split_sentences, stem, tokenize


@pytest.fixture(scope="module")
def articles():
    return load_articles()


@pytest.fixture(scope="module")
def index(articles):
    return BM25Index(articles)


def test_corpus_shape(articles):
    assert len(articles) == 25
    states = {a.workflow_state for a in articles}
    assert states == {"published", "draft", "retired"}


def test_tokenizer_and_stemmer():
    assert stem("printing") == "print"
    assert stem("printers") == "print"
    assert stem("policies") == "policy"
    assert tokenize("The printer is NOT printing!") == ["print", "print"]
    sentences = split_sentences("1. First step. Second part.\n- bullet")
    assert sentences == ["First step.", "Second part.", "bullet"]


@pytest.mark.parametrize("question,expected", [
    ("receipt printer at the teller counter is not printing", "KB0010007"),
    ("how do I renew an ssl certificate", "KB0010011"),
    ("MID server is down after upgrade", "KB0010012"),
    ("ACH batch rejected validation failed", "KB0010005"),
    ("database connection pool exhausted payments", "KB0010010"),
    ("fraud queue backlog", "KB0010019"),
])
def test_top_hit(index, question, expected):
    assert index.search(question)[0].article.number == expected


def test_unpublished_articles_are_not_searchable(articles, index):
    numbers = {a.number for a in index.articles}
    assert "KB0010024" not in numbers  # retired
    assert "KB0010025" not in numbers  # draft
    assert all(h.article.number != "KB0010025" for h in index.search("consumer lag streaming"))
    full = BM25Index(articles, include_unpublished=True)
    assert full.search("consumer lag streaming")[0].article.number == "KB0010025"


def test_title_matches_outrank_body_matches():
    docs = [Article("KB1", "Printer jam", "Paper."), Article("KB2", "Other", "printer mentioned once")]
    for d in docs:
        d.workflow_state = "published"
    hits = BM25Index(docs).search("printer")
    assert [h.article.number for h in hits] == ["KB1", "KB2"]


def test_answer_has_citations_and_sources(index):
    answer = KnowledgeAssistant(index).ask("card authorization is timing out")
    assert answer.answered and answer.mode == "extractive"
    assert answer.sources[0].number == "KB0010004"
    assert "[1]" in answer.text
    assert all(line.rstrip().endswith(("[1]", "[2]")) for line in answer.text.splitlines())
    assert answer.confidence == "high"


def test_answer_sentences_come_from_sources(index, articles):
    answer = KnowledgeAssistant(index).ask("my laptop is asking for a bitlocker key")
    bodies = " ".join(a.body for a in articles)
    for line in answer.text.splitlines():
        sentence = line[2:].rsplit(" [", 1)[0]
        assert sentence in bodies  # extractive: nothing invented


def test_abstains_when_out_of_scope(index):
    answer = KnowledgeAssistant(index).ask("what is the capital of France?")
    assert not answer.answered
    assert answer.sources == []
    assert "raise a ticket" in answer.text


def test_query_is_redacted(index):
    answer = KnowledgeAssistant(index).ask("customer 123456789012 locked out, mail x.y@contoso-bank.example")
    assert "123456789012" not in answer.query
    assert answer.redactions == {"ACCOUNT": 1, "EMAIL": 1}
    assert "redacted from question" in answer.render()


def test_is_grounded():
    assert is_grounded("Do this [1] and that [2].", 2)
    assert not is_grounded("Do this.", 2)
    assert not is_grounded("Do this [3].", 2)
    assert not is_grounded("NO_ANSWER", 2)
    assert not is_grounded("", 1)
