import pytest
from src.bgg_corpus.models.review import Review
from src.bgg_corpus.models.corpus_document import CorpusDocument

def test_corpus_document_structure():
    review = Review("u1", 9, "Loved it!", 1757887200)
    doc = CorpusDocument(review, processed={
        "tokens": ["loved", "it"],
        "language": "en",
        "patterns": {"emojis": ["😊"]}
    })

    result = doc.to_dict()
    assert "processed" in result
    assert result["language"] == "en"
    assert "tokens" in result["processed"]
    assert result["patterns"]["emojis"] == ["😊"]
