"""Runtime resources, language mappings, stopwords, and logging configuration."""

import logging
from nltk.corpus import stopwords
from ..config import API_DIR, CRAWLER_DIR, BALANCE_REPORTS_DIR

# -----------------------------
# Language mappings
# -----------------------------
NLTK_LANG_MAP = {
    "en": "english",
    "es": "spanish",
    "pt": "portuguese",
    "fr": "french",
    "de": "german",
    "it": "italian",
    "nl": "dutch"
} # Add more as needed

SPACY_MODELS = {}
SPACY_LANG_MAP = {
    "en": "en_core_web_sm",
    "es": "es_core_news_sm",
    "fr": "fr_core_news_sm",
    "de": "de_core_news_sm",
    "it": "it_core_news_sm",
    "pt": "pt_core_news_sm",
    "nl": "nl_core_news_sm"
} # Add more as needed

# -----------------------------
# Stopwords cache
# -----------------------------
STOPWORDS_CACHE = {}
for lang_code, nltk_name in NLTK_LANG_MAP.items():
    try:
        STOPWORDS_CACHE[nltk_name] = set(stopwords.words(nltk_name))
    except Exception:
        STOPWORDS_CACHE[nltk_name] = set()

# -----------------------------
# Paths (from config.py)
# -----------------------------
DATA_API_DIR = API_DIR  # adjust if you have a subfolder for API data
DATA_CRAWLER_DIR = CRAWLER_DIR
REPORTS_DIR = BALANCE_REPORTS_DIR

# -----------------------------
# Logger setup
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)


