"""Preprocessing pipeline for individual review items."""
import re

from .cleaning import normalize_text, extract_special_patterns
from .language import detect_language, get_spacy_lang_code, get_nltk_language
from .spacy_analysis import analyze_text_spacy
from .tokenization import apply_stemming
from ..features import LinguisticFeaturesExtractor
from ..resources import STOPWORDS_CACHE

def process_review_item(item, lower=True, remove_stopwords=True, correct_spelling=False, stem_method="porter"):
    """
    High-level preprocessing function for review items.
    Includes text normalization, language detection, tokenization, stemming, lemmatization and feature extraction.
    """
    raw = item.get("raw_text")
    if not raw or not raw.strip():
        return _empty_result(item, raw)

    # 1. Extract special patterns and normalize text
    special_patterns = extract_special_patterns(raw)
    clean = normalize_text(raw, lower=lower, correct_spelling=correct_spelling)

    # 2. Language detection
    detected_lang = detect_language(clean if clean else raw)
    spacy_lang = get_spacy_lang_code(detected_lang)
    nltk_lang = get_nltk_language(spacy_lang)
    stop_words_set = STOPWORDS_CACHE.get(nltk_lang, set())

    # 3. spaCy analysis: sentences, tokens, lemmas, POS, dependencies, entities
    sentences, tokens, lemmas, pos_tags, dependencies, entities = analyze_text_spacy(clean, detected_lang)

    # 4. Filter tokens: remove stopwords and punctuation
    tokens_no_stop = filter_tokens(tokens, stop_words_set, remove_stopwords=remove_stopwords)

    # 5. Stemming with NLTLK
    stems = apply_stemming(tokens_no_stop, spacy_lang, method=stem_method)

    # 6. Linguistic feature extraction
    extractor = LinguisticFeaturesExtractor()
    linguistic_features = extractor.extract_features(
        lemmas=lemmas,
        tokens_no_stopwords=tokens_no_stop,
        dependencies=dependencies,
        sentences=sentences,
        raw_text=raw
    )

    # 7. Assemble output
    return {
        **_base_result(item, raw, clean, detected_lang, spacy_lang),
        "sentences": sentences,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stop,
        "stems": stems,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        "dependencies": dependencies,
        "entities": entities,
        "linguistic_features": linguistic_features,
        "patterns": special_patterns
    }


def filter_tokens(tokens, stop_words_set, remove_stopwords=True):
    """
    Filter tokens by removing stopwords and non-alphanumeric tokens.
    Keeps only words with at least one alphanumeric character.
    """
    return [
        t for t in tokens
        if (not remove_stopwords or t.lower() not in stop_words_set)
        and re.search(r"\w", t)
    ]


def _empty_result(item, raw):
    return {
        **_base_result(item, raw, "", "unknown", "en"),
        "sentences": [],
        "tokens": [],
        "tokens_no_stopwords": [],
        "stems": [],
        "lemmas": [],
        "pos_tags": [],
        "dependencies": [],
        "entities": [],
        "linguistic_features": {},
        "patterns": {}
    }


def _base_result(item, raw, clean, detected_lang, spacy_lang):
    return {
        "username": item.get("username"),
        "rating": item.get("rating"),
        "timestamp": item.get("timestamp"),
        "raw_text": raw,
        "clean_text": clean,
        "language_detected": detected_lang,
        "language": spacy_lang,
    }