import pytest
import re
from unittest.mock import patch, MagicMock

# Imports
from src.bgg_corpus.preprocessing.cleaning import (
    normalize_text, extract_special_patterns, replace_thing_tags
)
from src.bgg_corpus.preprocessing.language.detection import detect_language
from src.bgg_corpus.preprocessing.language.spacy_utils import (
    get_nltk_language, get_spacy_lang_code, load_spacy_model_for
)
from src.bgg_corpus.preprocessing.spacy_analysis import analyze_text_spacy
from src.bgg_corpus.preprocessing.tokenization.stemming import apply_stemming


# ==========================================================
# 🧩 TESTS: cleaning.py
# ==========================================================

def test_replace_thing_tags_with_known_id():
    id2name = {123: "Catan"}
    text = "I love [thing=123][/thing]!"
    assert replace_thing_tags(text, id2name) == "I love Catan!"

def test_replace_thing_tags_with_unknown_id():
    id2name = {}
    text = "I love [thing=999][/thing]!"
    assert replace_thing_tags(text, id2name) == "I love !"

def test_normalize_text_basic_cleaning():
    raw = "This is a <b>great</b> game! Visit https://bgg.com for info."
    cleaned = normalize_text(raw)
    assert "<b>" not in cleaned
    assert "https" not in cleaned
    assert "great" in cleaned
    assert "visit" in cleaned

def test_normalize_text_lower_false():
    raw = "Catan is GREAT!!!"
    cleaned = normalize_text(raw, lower=False)
    assert "GREAT" in cleaned
    assert "great" not in cleaned

def test_normalize_text_spell_correction(monkeypatch):
    """Simula corrección ortográfica sin usar TextBlob real."""
    from src.bgg_corpus.preprocessing import cleaning
    monkeypatch.setattr(cleaning, "TextBlob", lambda x: type("MockBlob", (), {"correct": lambda self: "fixed text"})())
    result = normalize_text("some txt", correct_spelling=True)
    assert result == "fixed text"

def test_extract_special_patterns_all():
    text = (
        "Contact me at test@example.com on 12/05/2024. "
        "Call +34 678 123 456. Visit https://bgg.com #boardgames @user 😀"
    )
    patterns = extract_special_patterns(text)
    assert patterns["emails"] == ["test@example.com"]
    assert patterns["dates"] == ["12/05/2024"]
    assert "+34" in patterns["phones"][0]
    assert patterns["hashtags"] == ["#boardgames"]
    assert patterns["mentions"] == ["@user"]
    assert len(patterns["urls"]) == 1
    assert any(e["emoji"] == "😀" for e in patterns["emojis"])


# ==========================================================
# 🧩 TESTS: language/detection.py
# ==========================================================

def test_detect_language_valid_text():
    result = detect_language("Este es un texto en español.")
    assert result in ["es", "unknown"]

def test_detect_language_empty_text():
    assert detect_language("") == "unknown"

def test_detect_language_low_confidence(monkeypatch):
    import src.bgg_corpus.preprocessing.language.detection as ld
    monkeypatch.setattr(ld.langid, "classify", lambda x: ("en", 0.2))
    assert ld.detect_language("hello") == "unknown"


# ==========================================================
# 🧩 TESTS: language/spacy_utils.py
# ==========================================================

def test_get_nltk_language_default():
    assert get_nltk_language("xx") == "english"

def test_get_spacy_lang_code_normalization():
    assert get_spacy_lang_code("es-ES") == "es"
    assert get_spacy_lang_code("unknown") == "en"

@patch("src.bgg_corpus.preprocessing.language.spacy_utils.spacy.load")
def test_load_spacy_model_for_success(mock_load):
    mock_model = MagicMock()
    mock_load.return_value = mock_model
    from src.bgg_corpus.preprocessing.language.spacy_utils import SPACY_LANG_MAP
    SPACY_LANG_MAP["en"] = ["en_core_web_sm"]
    result = load_spacy_model_for("en")
    assert result == mock_model

@patch("src.bgg_corpus.preprocessing.language.spacy_utils.spacy.load", side_effect=Exception("fail"))
def test_load_spacy_model_for_failure(mock_load):
    from src.bgg_corpus.preprocessing.language.spacy_utils import SPACY_LANG_MAP
    SPACY_LANG_MAP["en"] = ["nonexistent_model"]
    result = load_spacy_model_for("en")
    assert result is None


# ==========================================================
# 🧩 TESTS: spacy_analysis.py
# ==========================================================

@patch("src.bgg_corpus.preprocessing.spacy_analysis.load_spacy_model_for")
@patch("src.bgg_corpus.preprocessing.spacy_analysis.get_spacy_lang_code")
def test_analyze_text_spacy_basic(mock_code, mock_load):
    mock_code.return_value = "en"

    # Crear un mock de nlp y doc
    mock_token = MagicMock()
    mock_token.text = "Hello"
    mock_token.lemma_ = "hello"
    mock_token.pos_ = "NOUN"
    mock_token.tag_ = "NN"
    mock_token.dep_ = "ROOT"
    mock_token.head.text = "Hello"
    mock_token.is_space = False
    mock_token.is_punct = False

    mock_sent = MagicMock()
    mock_sent.text = "Hello world."
    mock_doc = MagicMock(sents=[mock_sent], __iter__=lambda self: iter([mock_token]), ents=[])

    mock_load.return_value = lambda text: mock_doc

    result = analyze_text_spacy("Hello world", "en", stop_words_set={"hello"})
    sentences, tokens, tokens_no_stop, lemmas, pos_tags, deps, ents = result

    assert "Hello world." in sentences
    assert tokens == ["hello"]
    assert lemmas == ["hello"]
    assert ("Hello", "ROOT", "Hello") in deps


# ==========================================================
# 🧩 TESTS: tokenization/stemming.py
# ==========================================================

def test_apply_stemming_basic():
    tokens = ["playing", "games"]
    result = apply_stemming(tokens, "en", method="snowball")
    assert all(isinstance(t, str) for t in result)
    assert len(result) == len(tokens)

def test_apply_stemming_empty_list():
    assert apply_stemming([], "en") == []