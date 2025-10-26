import langid
from ...resources import NLTK_LANG_MAP

langid.set_languages(list(NLTK_LANG_MAP.keys()))

def detect_language(text, min_confidence=0.7):
    """
    Detect language using langid. Returns ISO 639-1 code or 'unknown'.
    """
    if not text or not text.strip():
        return "unknown"
    text = text.replace("\n", " ").strip()
    code, confidence = langid.classify(text)
    return code if confidence >= min_confidence else "unknown"