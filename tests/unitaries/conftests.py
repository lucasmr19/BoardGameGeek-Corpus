import pytest
from unittest.mock import MagicMock

# ==========================================================
# 📌 Fixture: textos de ejemplo
# ==========================================================

@pytest.fixture
def sample_text_html_bbcode():
    return "This is a <b>bold</b> and [i]italic[/i] review. Visit https://bgg.com!"

@pytest.fixture
def sample_text_with_patterns():
    return (
        "Contact me at test@example.com on 12/05/2024. "
        "Call +34 678 123 456. Visit https://bgg.com #boardgames @user 😀"
    )

@pytest.fixture
def sample_text_thing_tags():
    return "I love [thing=123][/thing] and [thing=999][/thing]!"

@pytest.fixture
def id2name_mapping():
    return {123: "Catan"}

# ==========================================================
# 📌 Fixture: mock de TextBlob
# ==========================================================

@pytest.fixture
def mock_textblob(monkeypatch):
    class MockTextBlob:
        def __init__(self, text):
            self.text = text

        def correct(self):
            return "corrected text"

    monkeypatch.setattr("src.bgg_corpus.preprocessing.cleaning.TextBlob", MockTextBlob)
    return MockTextBlob

# ==========================================================
# 📌 Fixture: mock de spaCy nlp y tokens
# ==========================================================

@pytest.fixture
def mock_spacy_doc():
    """Crea un doc simulado con un token y una oración."""
    # Token simulado
    token = MagicMock()
    token.text = "Hello"
    token.lemma_ = "hello"
    token.pos_ = "NOUN"
    token.tag_ = "NN"
    token.dep_ = "ROOT"
    token.head.text = "Hello"
    token.is_space = False
    token.is_punct = False

    # Sentencia simulada
    sent = MagicMock()
    sent.text = "Hello world."

    # Doc simulado
    doc = MagicMock()
    doc.sents = [sent]
    doc.__iter__ = lambda self: iter([token])
    doc.ents = []

    return doc

@pytest.fixture
def mock_spacy_model(mock_spacy_doc):
    """Crea un objeto nlp simulado que devuelve el doc mock."""
    def nlp(text):
        return mock_spacy_doc
    return nlp
