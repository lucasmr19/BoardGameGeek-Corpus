import langid
from ...resources import NLTK_LANG_MAP, LOGGER

langid.set_languages(list(NLTK_LANG_MAP.keys()))

def detect_language(text, min_confidence=0.85, default_lang="en", context=None):
    """
    Detect language using langid. Returns ISO 639-1 code or fallback (default_lang).
    If `context` is provided (dict or object), include metadata when text is invalid or empty.
    """
    if not text or not text.strip():
        if context:
            # try to print as much as possible about the review
            try:
                ctx_info = {
                    "username": getattr(context, "username", None) or context.get("username"),
                    "rating": getattr(context, "rating", None) or context.get("rating"),
                    "timestamp": getattr(context, "timestamp", None) or context.get("timestamp"),
                    "raw_text": getattr(context, "comment", None) or context.get("raw_text"),
                    "clean_text": getattr(context, "clean_text", None) or context.get("clean_text"),
                    "game_id": getattr(context, "game_id", None) or context.get("game_id"),
                }
                LOGGER.warning(
                    "[LanguageDetect] Empty or whitespace-only text detected → fallback to '%s'\n"
                    "→ Context: username=%s | rating=%s | timestamp=%s | game_id=%s \n" 
                    "→ raw_text='%s'\n→ clean_text='%s'",
                    default_lang,
                    ctx_info["username"],
                    ctx_info["rating"],
                    ctx_info["timestamp"],
                    ctx_info["game_id"],
                    (ctx_info["raw_text"] or "").strip()[:200],
                    (ctx_info["clean_text"] or "").strip()[:200],
                )
            except Exception as e:
                LOGGER.warning("[LanguageDetect] Failed to print context info: %s", e)
        else:
            LOGGER.warning("[LanguageDetect] Empty or whitespace-only text detected → fallback to '%s'", default_lang)

        return default_lang

    text = text.replace("\n", " ").strip()
    preview = (text[:80] + "...") if len(text) > 80 else text

    try:
        code, confidence = langid.classify(text)
        LOGGER.debug(f"[LanguageDetect] Detected '{code}' (confidence={confidence:.3f}) for text='{preview}'")

        if confidence < min_confidence:
            return default_lang
        return code

    except Exception as e:
        LOGGER.warning(f"[LanguageDetect] Exception during detection for text='{preview}' | Error: {e}")
        return default_lang