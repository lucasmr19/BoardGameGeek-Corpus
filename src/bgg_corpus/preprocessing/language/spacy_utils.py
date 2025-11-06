import os
import spacy
from ...resources import NLTK_LANG_MAP, SPACY_MODELS, SPACY_LANG_MAP, LOGGER

def get_nltk_language(code):
    """Map ISO code to NLTK stopwords language."""
    return NLTK_LANG_MAP.get(code, "english")


def get_spacy_lang_code(code):
    """Normalize ISO code to base spaCy language code (default: 'en')."""
    if not code or code == "unknown":
        return "en"
    base = code.split("-")[0].lower()
    return base if base in SPACY_LANG_MAP else "en"

def init_spacy_models():
    """Initializer for multiprocessing — preload spaCy models once per worker."""
    for code, candidates in SPACY_LANG_MAP.items():
        if code not in SPACY_MODELS or SPACY_MODELS[code] is None:
            for model_name in candidates:
                try:
                    SPACY_MODELS[code] = spacy.load(model_name)
                    LOGGER.info(f"[Worker {os.getpid()}] Loaded spaCy model: {model_name}")
                    break
                except Exception:
                    continue


def load_spacy_model_for(code):
    """
    Load and cache a spaCy model for the given ISO language code.
    Always loads with NER, POS, parser, and lemmatizer enabled.
    """
    if code in SPACY_MODELS and SPACY_MODELS[code] is not None:
        return SPACY_MODELS[code]

    model_candidates = SPACY_LANG_MAP.get(code)
    if not model_candidates:
        SPACY_MODELS[code] = None
        return None

    for model_name in model_candidates:
        try:
            nlp = spacy.load(model_name)
            SPACY_MODELS[code] = nlp
            LOGGER.info(f"Loaded spaCy model: {model_name}")
            return nlp
        except Exception as e:
            LOGGER.warning(f"Could not load spaCy model {model_name}: {e}")

    SPACY_MODELS[code] = None
    return None