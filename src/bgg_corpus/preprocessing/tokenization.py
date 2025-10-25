"""
Tokenization utilities for text preprocessing from the BoardGameGeek website.
Includes sentence segmentation, tokenization, stopword removal, and stemming.
"""

import re
import os
import nltk
from langdetect import DetectorFactory
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, SnowballStemmer

from ..resources import NLTK_LANG_MAP, LOGGER


# ---------------- CONFIG ----------------
DetectorFactory.seed = 0

nltk_data_dir = os.path.join(os.path.expanduser("~"), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

#for pkg in ("punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab", "averaged_perceptron_tagger_eng"):
#    try:
#        nltk.data.find(f"tokenizers/{pkg}") if "punkt" in pkg else nltk.data.find(f"corpora/{pkg}")
#    except LookupError:
#        nltk.download(pkg, download_dir=nltk_data_dir, quiet=True)

def sentence_segmentation(text, nltk_lang):
    if not text or not text.strip():
        return []
    
    try:
        sentences = sent_tokenize(text, language=nltk_lang)
    except Exception as e:
        LOGGER.warning(f"sent_tokenize failed: {e}. Using regex fallback.")
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    
    return [s for s in sentences if s.strip()]

def tokenize_and_filter(text, nltk_lang, stop_words_set, remove_stopwords=True):
    if not text or not text.strip():
        return [], []
    
    try:
        tokens = word_tokenize(text, language=nltk_lang)
    except Exception as e:
        LOGGER.warning(f"word_tokenize failed: {e}. Using regex fallback.")
        tokens = re.findall(r"\w+", text)
    
    tokens_alpha = [t for t in tokens if re.search(r"\w", t)]
    
    if remove_stopwords:
        tokens_no_stop = [t for t in tokens_alpha if t.lower() not in stop_words_set]
    else:
        tokens_no_stop = tokens_alpha
    
    return tokens_alpha, tokens_no_stop

def apply_stemming(tokens, spacy_lang):
    if not tokens:
        return []
    
    lang = NLTK_LANG_MAP.get(spacy_lang, "english")
    
    if lang == "english":
        stemmer = PorterStemmer()
    else:
        try:
            stemmer = SnowballStemmer(lang)
        except Exception:
            stemmer = SnowballStemmer("english")
    
    return [stemmer.stem(t) for t in tokens]