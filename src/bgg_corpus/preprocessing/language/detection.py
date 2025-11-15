"""
Module to detect language using a transformer-based model (XLM-RoBERTa).

Provides a high-accuracy language detection method supporting 100 languages.
"""

import torch
from transformers import pipeline
from ...resources import LOGGER

_lang_detector = None

def get_language_detector():
    """
    Lazy load the XLM-RoBERTa language detection model.
    
    This ensures the model is loaded only once and reused for subsequent calls.
    
    Returns:
        transformers.Pipeline: HuggingFace pipeline for text classification.
    """
    global _lang_detector
    if _lang_detector is None:
        try:
            # Use GPU if available, otherwise CPU
            device = 0 if torch.cuda.is_available() else -1
            _lang_detector = pipeline(
                "text-classification",
                model="papluca/xlm-roberta-base-language-detection",
                device=device
            )
            LOGGER.info(f"[LanguageDetect] XLM-RoBERTa loaded (device={'GPU' if device == 0 else 'CPU'})")
        except Exception as e:
            LOGGER.error(f"[LanguageDetect] Failed to load XLM-RoBERTa model: {e}")
            raise
    return _lang_detector


def detect_language(text, min_confidence=0.70, default_lang="en", context=None):
    """
    Detect the language of a given text using XLM-RoBERTa.

    The function returns an ISO 639-1 language code (e.g., 'en', 'fr', 'es').
    If the text is empty or detection confidence is below `min_confidence`, 
    it returns `default_lang`.

    Args:
        text (str): The text to analyze.
        min_confidence (float, optional): Minimum confidence threshold to accept the prediction. Default is 0.70.
        default_lang (str, optional): Fallback language code if detection fails. Default is 'en'.
        context (dict or object, optional): Additional context for logging purposes (e.g., username, rating, game_id).

    Returns:
        str: ISO 639-1 language code detected or `default_lang` as fallback.
    """
    if not text or not text.strip():
        if context:
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
                    "[LanguageDetect] Empty text → fallback to '%s'\n"
                    "→ Context: username=%s | rating=%s | timestamp=%s | game_id=%s\n"
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
                LOGGER.warning("[LanguageDetect] Failed to print context: %s", e)
        else:
            LOGGER.warning("[LanguageDetect] Empty text → fallback to '%s'", default_lang)
        return default_lang

    # Normalize text for detection
    text_clean = text.replace("\n", " ").strip()
    preview = (text_clean[:80] + "...") if len(text_clean) > 80 else text_clean

    try:
        detector = get_language_detector()
        
        # Truncate long texts (model limit ~512 tokens)
        if len(text_clean) > 2000:
            text_clean = text_clean[:2000]
        
        result = detector(text_clean, top_k=1)[0]
        code = result['label']
        confidence = result['score']
        
        LOGGER.debug(
            f"[LanguageDetect] Detected '{code}' (confidence={confidence:.3f}) "
            f"for text='{preview}'"
        )

        if confidence < min_confidence:
            LOGGER.debug(
                f"[LanguageDetect] Low confidence ({confidence:.3f} < {min_confidence}) "
                f"→ fallback to '{default_lang}'"
            )
            return default_lang
        
        return code

    except Exception as e:
        LOGGER.warning(
            f"[LanguageDetect] Exception for text='{preview}' | Error: {e}"
        )
        return default_lang
