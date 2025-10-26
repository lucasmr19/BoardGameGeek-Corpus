"""Module for applying stemming to tokens using various stemming algorithms."""
from nltk.stem import SnowballStemmer

from .tokenizers_registry import STEMMERS
from ...resources import NLTK_LANG_MAP, LOGGER

def apply_stemming(tokens, spacy_lang, method="porter"):
    if not tokens:
        return []

    lang = NLTK_LANG_MAP.get(spacy_lang, "english")

    try:
        if method == "snowball":
            stemmer = SnowballStemmer(lang)
        if method in STEMMERS:
            stemmer_cls = STEMMERS[method]
            stemmer = stemmer_cls() if callable(stemmer_cls) else stemmer_cls(lang)
        else:
            LOGGER.error(f"Unknown stemmer '{method}', using Porter.")
    except Exception as e:
        LOGGER.error(f"Stemmer '{method}' failed: {e}. Using Porter fallback.")

    return [stemmer.stem(t) for t in tokens]