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
import os
import html
from textblob import TextBlob
import pandas as pd
import emoji
import unidecode

from ..config import RANKS_DF
from ..resources import LOGGER

# ---------------- REGEX ----------------
EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
DATE_RE = re.compile(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b")
PHONE_RE = re.compile(r"\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{4,10}")
HASHTAG_RE = re.compile(r"#\w+")
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

ABBREVIATIONS = {
    "Sr": "Señor", "Sra": "Señora", "Dr": "Doctor", "Dra": "Doctora",
    "EE.UU": "Estados Unidos", "etc": "etcétera", "info": "information",
    "mins": "minutes", "hr": "hour", "yrs": "years"
}#... Add more as needed

try:
    id2name = dict(zip(RANKS_DF["id"], RANKS_DF["name"]))
except FileNotFoundError:
    id2name = {}
    LOGGER.warning(f"{RANKS_DF} not found. Thing tag replacement disabled.")

def replace_thing_tags(text, id2name):
    def repl(match):
        game_id = int(match.group(1))
        return id2name.get(game_id, "")
    return re.sub(r"\[thing=(\d+)\]\[\/thing\]", repl, text)

def normalize_text(text, lower=True, correct_spelling=False):
    if not text:
        return ""
    
    text = html.unescape(text)
    text = replace_thing_tags(text, id2name)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[/?[ib]\]", " ", text)
    text = URL_RE.sub(" ", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = text.replace("&", " and ")
    text = re.sub(r"[\t\v\f]+", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"([!?])\1{1,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\.{2,}", ".", text)
    
    if lower:
        text = text.lower()
    
    text = text.replace("'", "'")
    text = unidecode.unidecode(text)
    text = emoji.demojize(text, delimiters=(":", ":"))
    text = re.sub(r':[a-zA-Z0-9_]+:', '', text)
    text = re.sub(r'[:;=8][-~]?[)(DPpOo]', '', text)
    
    if correct_spelling:
        text = str(TextBlob(text).correct())
    
    words = text.split()
    words = [ABBREVIATIONS.get(w, w) for w in words]
    text = " ".join(words)
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def extract_special_patterns(text):
    return {
        "emails": EMAIL_RE.findall(text),
        "dates": DATE_RE.findall(text),
        "phones": PHONE_RE.findall(text),
        "hashtags": HASHTAG_RE.findall(text),
        "mentions": MENTION_RE.findall(text),
        "urls": URL_RE.findall(text),
        "emojis": [c for c in text.split() if c.startswith(":") and c.endswith(":")],
    }