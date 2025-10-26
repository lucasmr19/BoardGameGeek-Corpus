from .detection import detect_language
from .spacy_utils import get_nltk_language, get_spacy_lang_code, load_spacy_model_for

__all__ = [
    "detect_language",
    "get_nltk_language",
    "get_spacy_lang_code",
    "load_spacy_model_for",
]