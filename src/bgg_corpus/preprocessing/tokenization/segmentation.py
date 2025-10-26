"""Module for sentence segmentation using various tokenization methods."""

from .tokenizers_registry import SENT_TOKENIZERS
from ...resources import LOGGER

def sentence_segmentation(text, nltk_lang, method="punkt"):
    if not text or not text.strip():
        return []
    if method not in SENT_TOKENIZERS:
        LOGGER.warning(f"Unknown sentence tokenizer '{method}', using regex fallback.")
        method = "regex"

    try:
        tokenizer = SENT_TOKENIZERS[method](nltk_lang)
        sentences = tokenizer(text)
    except Exception as e:
        LOGGER.warning(f"Sentence tokenizer '{method}' failed: {e}. Using regex fallback.")
        sentences = SENT_TOKENIZERS["regex"](nltk_lang)(text)
    
    return [s for s in sentences if s.strip()]