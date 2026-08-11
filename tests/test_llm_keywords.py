import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.keywords import _verify


def test_verify_keeps_real_citations_and_recomputes_count():
    phrases = [{"phrase": "charged twice", "mention_count": 5, "review_ids": ["1", "2"],
               "example_quote": "..."}]
    out = _verify(phrases, valid_ids={"1", "2", "3"}, corpus_size=10)
    assert out[0]["term"] == "charged twice"
    assert out[0]["example_review_ids"] == ["1", "2"]
    assert out[0]["count"] == 2  # recomputed from verified ids, not trusted as 5
    assert out[0]["share_of_corpus"] == 20.0  # 2/10
    assert out[0]["hallucinated_citations_dropped"] == 0


def test_verify_drops_hallucinated_ids_and_recomputes_count():
    phrases = [{"phrase": "charged twice", "mention_count": 3, "review_ids": ["1", "999"],
               "example_quote": "..."}]
    out = _verify(phrases, valid_ids={"1", "2"}, corpus_size=4)
    assert out[0]["example_review_ids"] == ["1"]
    assert out[0]["count"] == 1
    assert out[0]["hallucinated_citations_dropped"] == 1


def test_verify_drops_phrase_with_zero_real_citations():
    phrases = [{"phrase": "made up complaint", "mention_count": 4, "review_ids": ["999", "888"],
               "example_quote": "..."}]
    out = _verify(phrases, valid_ids={"1", "2"}, corpus_size=2)
    assert out == []


def test_verify_sorts_by_recomputed_count_descending():
    phrases = [
        {"phrase": "a", "mention_count": 10, "review_ids": ["1"], "example_quote": ""},        # -> 1 real
        {"phrase": "b", "mention_count": 1, "review_ids": ["1", "2", "3"], "example_quote": ""},  # -> 3 real
    ]
    out = _verify(phrases, valid_ids={"1", "2", "3"}, corpus_size=3)
    assert [p["term"] for p in out] == ["b", "a"]


def test_verify_zero_corpus_size_does_not_divide_by_zero():
    phrases = [{"phrase": "a", "mention_count": 1, "review_ids": ["1"], "example_quote": ""}]
    out = _verify(phrases, valid_ids={"1"}, corpus_size=0)
    assert out[0]["share_of_corpus"] == 0.0
