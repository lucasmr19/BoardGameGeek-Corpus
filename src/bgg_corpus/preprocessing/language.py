"""
language.py

Contains functions for language detection and linguistic analysis.

- Uses langid for robust language detection on short or mixed-language text.
- Falls back to spaCy for tokenization, POS tagging, dependencies, and lemmatization.
- If spaCy is not available for a language, uses NLTK WordNet lemmatizer for English,
  or SnowballStemmer for other languages as a fallback.
- Handles unknown or empty text gracefully.
"""

import re
import os
import nltk
import langid
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import spacy

from ..resources import NLTK_LANG_MAP, SPACY_MODELS, SPACY_LANG_MAP, LOGGER

# ---------------- CONFIG ----------------
nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Set seed for reproducibility
langid.set_languages([code for code in NLTK_LANG_MAP.keys()])  # Optional: restrict to supported languages

def detect_language(text, min_confidence=0.7):
    """
    Detect the language of the given text using langid.
    Returns ISO 639-1 code if confident, otherwise 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"
    text = text.replace("\n", " ").strip()
    code, confidence = langid.classify(text)
    return code if confidence >= min_confidence else "unknown"

def get_nltk_language(code):
    """Map ISO code to NLTK language name."""
    return NLTK_LANG_MAP.get(code, "english")

def get_spacy_lang_code(detected_code):
    """Map detected ISO code to spaCy language code."""
    if not detected_code or detected_code == "unknown":
        return "en"
    base = detected_code.split("-")[0].lower()
    return base if base in SPACY_LANG_MAP else "en"

def load_spacy_model_for(code):
    """Load spaCy model for a given language code, caching it in SPACY_MODELS."""
    if code in SPACY_MODELS and SPACY_MODELS[code] is not None:
        return SPACY_MODELS[code]

    model_name = SPACY_LANG_MAP.get(code)
    if not model_name:
        SPACY_MODELS[code] = None
        return None

    try:
        nlp = spacy.load(model_name, disable=["ner"])
        SPACY_MODELS[code] = nlp
        return nlp
    except Exception as e:
        SPACY_MODELS[code] = None
        LOGGER.warning(f"spaCy model {model_name} not available: {e}")
        return None

def analyze_text_spacy(text, detected_code):
    """
    Analyze text using spaCy or fallback to NLTK/Snowball stemmer.
    Returns a dict with:
        - spacy_used: bool
        - pos_tags: list of (word, POS, tag)
        - dependencies: list of (word, dependency, head)
        - lemmas: list of lemmas/stems
    """
    code = get_spacy_lang_code(detected_code)
    nlp = load_spacy_model_for(code)

    if nlp:
        doc = nlp(text)
        pos_tags = [(token.text, token.pos_, token.tag_) for token in doc]
        dependencies = [(token.text, token.dep_, token.head.text) for token in doc]
        lemmas = [token.lemma_ for token in doc if not token.is_stop and token.lemma_.strip()]
        return {"spacy_used": True, "pos_tags": pos_tags, "dependencies": dependencies, "lemmas": lemmas}

    # Fallback for English using NLTK WordNet lemmatizer
    if code == "en":
        lemmatizer = WordNetLemmatizer()
        tokens = re.findall(r"\w+", text.lower())
        lemmas = [lemmatizer.lemmatize(t) for t in tokens if t.strip()]
        return {"spacy_used": False, "pos_tags": [], "dependencies": [], "lemmas": lemmas}

    # Fallback for other languages using Snowball stemmer
    try:
        stemmer = SnowballStemmer(code)
    except Exception:
        stemmer = SnowballStemmer("english")
    tokens = re.findall(r"\w+", text.lower())
    pseudo_lemmas = [stemmer.stem(t) for t in tokens if t.strip()]
    return {"spacy_used": False, "pos_tags": [], "dependencies": [], "lemmas": pseudo_lemmas}
