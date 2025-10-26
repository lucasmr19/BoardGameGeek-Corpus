"""
Contains functions for cleaning and normalizing text data from the BoardGameGeek website.

Preprocessing includes:

1. HTML and formatting normalization:
   - Convert HTML entities to characters (e.g., &amp; → &).
   - Remove HTML tags (`<...>`).
   - Remove BBCode formatting tags like `[i]` and `[b]`.

2. Thing tag replacement:
   - Replace `[thing=<id>][/thing]` tags with actual game names using a lookup table.
   - If the game ID is not found, the tag is removed.

3. URL, mention, and social pattern removal:
   - Remove URLs, hashtags, mentions, phone numbers, and emails using regex.
   - Extract these patterns for separate analysis if needed.

4. Whitespace and punctuation normalization:
   - Replace line breaks and carriage returns with spaces.
   - Normalize repeated characters (e.g., "soooo" → "soo").
   - Normalize repeated punctuation (e.g., "!!!" → "!!").
   - Collapse multiple spaces into a single space.
   - Normalize repeated periods (e.g., "..." → ".").

5. Lowercasing (optional):
   - Convert all text to lowercase.

6. Character normalization:
   - Convert accented characters to ASCII equivalents (e.g., "é" → "e").
   - Replace emojis with their text descriptions using `emoji.demojize`.
   - Remove remaining emoji placeholders and common emoticons.

7. Special token handling:
   - Replace known abbreviations with full forms (e.g., "Dr" → "Doctor", "EE.UU" → "Estados Unidos").
   - Remove unwanted extra whitespace after replacements.

8. Optional spelling correction:
   - Use `TextBlob` to correct misspelled words if `correct_spelling=True`. This step may be skipped for performance.

9. Pattern extraction (`extract_special_patterns`):
   - Extract and return a dictionary of special tokens from text:
     - Emails
     - Dates (format DD/MM/YYYY or DD-MM-YYYY)
     - Phone numbers
     - Hashtags
     - Mentions
     - URLs
     - Emojis
"""


import re
import html
import emoji
import unidecode
from textblob import TextBlob

from ..config import RANKS_DF
from ..resources import LOGGER


# ---------------- REGEX ----------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
DATE_RE = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
PHONE_RE = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{4,10}")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(
    r"""(?xi)
    (?:
        (?:https?://|www\d{0,3}[.])                     # http://, https://, or www.
        (?:[^\s()<>{}\[\]]+|(?:\([^\s()<>]+\)))+       # domain and path
        (?:\([^\s()<>]+\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])*
    )
    """,
    re.VERBOSE,
)

ABBREVIATIONS = {
    "Sr": "Señor", "Sra": "Señora", "Dr": "Doctor", "Dra": "Doctora",
    "EE.UU": "Estados Unidos", "etc": "etcétera", "info": "information",
    "mins": "minutes", "hr": "hour", "yrs": "years"
} # Add more as needed

# ---------------- LOAD RANK MAPPING ----------------
try:
    id2name = dict(zip(RANKS_DF["id"], RANKS_DF["name"]))
except Exception:
    id2name = {}
    LOGGER.warning(f"{RANKS_DF} not found. Thing tag replacement disabled.")


# ---------------- HELPERS ----------------
def replace_thing_tags(text, id2name):
    """Replace [thing=id][/thing] tags with game names."""
    return re.sub(
        r"\[thing=(\d+)\]\[\/thing\]",
        lambda m: id2name.get(int(m.group(1)), ""),
        text
    )


# ---------------- MAIN CLEANING ----------------
def normalize_text(text, lower=True, correct_spelling=False):
    """Normalize and clean a review text string."""
    if not text:
        return ""

    # --- HTML & BBCode cleanup ---
    text = html.unescape(text)
    text = replace_thing_tags(text, id2name)
    text = re.sub(r"<[^>]+>", " ", text)  # remove HTML tags
    text = re.sub(r"\[/?[a-zA-Z]+\]", " ", text)  # [b], [i], etc.

    # --- Remove URLs early ---
    text = URL_RE.sub(" ", text)

    # --- Normalize whitespace and punctuation ---
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[\t\v\f]+", " ", text)
    text = re.sub(r"&", " and ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)     # limit letter repetition
    text = re.sub(r"([!?])\1{1,}", r"\1\1", text)  # limit punctuation repetition
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"\s+", " ", text).strip()

    if lower:
        text = text.lower()

    # --- Normalize text (latin only, emojis -> text placeholders) ---
    text = unidecode.unidecode(text)
    text = emoji.demojize(text, delimiters=(":", ":"))
    text = re.sub(r':[a-zA-Z0-9_]+:', '', text)      # remove :emoji_names:
    text = re.sub(r'[:;=8][-~]?[)(DPpOo]', '', text) # remove text emoticons

    # --- Optional spell correction ---
    if correct_spelling:
        text = str(TextBlob(text).correct())

    # --- Expand abbreviations ---
    words = [ABBREVIATIONS.get(w, w) for w in text.split()]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------- PATTERN EXTRACTION ----------------
def extract_special_patterns(text):
    """Extract emails, dates, phones, hashtags, mentions, URLs, and emojis."""
    urls = URL_RE.findall(text)
    return {
        "emails": EMAIL_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "phones": PHONE_RE.findall(text),
        "hashtags": HASHTAG_RE.findall(text),
        "mentions": MENTION_RE.findall(text),
        "urls": urls,
        "emojis": emoji.emoji_list(text),
    }
