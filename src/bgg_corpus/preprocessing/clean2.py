"""
Limpieza y normalización de texto SIN eliminar acentos.
"""
import re
import html
import emoji
from textblob import TextBlob
from typing import List

from ..models import GameCorpus
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
        (?:https?://|www\d{0,3}[.])
        (?:[^\s()<>{}\[\]]+|(?:\([^\s()<>]+\)))+
        (?:\([^\s()<>]+\)|[^\s`!()\[\]{};:'".,<>?«»""''])*
    )
    """,
    re.VERBOSE,
)

ABBREVIATIONS = {
    "Sr": "Señor", "Sra": "Señora", "Dr": "Doctor", "Dra": "Doctora",
    "EE.UU": "Estados Unidos", "etc": "etcétera", "info": "information",
    "mins": "minutes", "hr": "hour", "yrs": "years"
}

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


def remove_only_problematic_unicode(text):
    """
    Elimina solo caracteres problemáticos (control chars, zero-width, etc.)
    MANTIENE acentos y caracteres especiales de idiomas.
    """
    # Remover caracteres de control (excepto espacios normales)
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    # Remover zero-width chars y otros invisibles problemáticos
    text = re.sub(r'[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]', '', text)
    
    # Normalizar comillas y guiones raros
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' '
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # " "
    text = text.replace('\u2013', '-').replace('\u2014', '-')  # – —
    text = text.replace('\u2026', '...')  # …
    
    return text


# ---------------- MAIN CLEANING ----------------
def normalize_text(text, lower=True, correct_spelling=False):
    """
    Normaliza texto de reviews PRESERVANDO acentos y caracteres especiales.
    
    Args:
        text: Texto original
        lower: Si convertir a minúsculas
        correct_spelling: Si aplicar corrección ortográfica (lento)
    
    Returns:
        Texto normalizado
    """
    if not text or not text.strip():
        return ""

    # --- HTML & BBCode cleanup ---
    text = html.unescape(text)
    text = replace_thing_tags(text, id2name)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\[/?[a-zA-Z]+\]", " ", text)

    # --- Remove URLs early ---
    text = URL_RE.sub("URL", text)

    # --- Normalize whitespace ---
    text = text.replace("\r", " ").replace("\n", " ")
    text = re.sub(r"[\t\v\f]+", " ", text)
    
    # --- Reduce repeated chars (pero NO para acentos) ---
    # Solo para caracteres ASCII básicos
    text = re.sub(r"([a-zA-Z0-9])\1{2,}", r"\1\1", text)
    text = re.sub(r"([!?])\1{1,}", r"\1\1", text)
    text = re.sub(r"\.{2,}", ".", text)
    
    # --- Consolidar espacios ---
    text = re.sub(r"\s+", " ", text).strip()

    # --- Convertir a minúsculas SI se solicita ---
    if lower:
        text = text.lower()

    # --- Convert emojis to text (ESTO SÍ, los emojis no son palabras) ---
    text = emoji.demojize(text, delimiters=(" ", " "))

    # --- CRÍTICO: NO usar unidecode - Eliminar solo chars problemáticos ---
    text = remove_only_problematic_unicode(text)

    # --- Optional spelling correction ---
    if correct_spelling:
        try:
            text = str(TextBlob(text).correct())
        except Exception as e:
            LOGGER.debug(f"Spelling correction failed: {e}")

    # --- Expand abbreviations (solo en inglés/español) ---
    # Cuidado: esto puede ser problemático para otros idiomas
    words = text.split()
    expanded_words = []
    for w in words:
        # Solo expandir si NO tiene acentos (probablemente inglés/español formal)
        if re.match(r'^[a-zA-Z]+$', w):
            expanded_words.append(ABBREVIATIONS.get(w, w))
        else:
            expanded_words.append(w)
    text = " ".join(expanded_words).strip()

    return text


# ---------------- VALIDACIÓN -------------------
def validate_cleaning():
    """Test cases para verificar que los acentos se preservan."""
    test_cases = [
        ("c'est incroyable! très intéressant", 
         "c'est incroyable! très intéressant"),
        
        ("¡Hola! ¿Cómo estás? ñoño español", 
         "hola cómo estás ñoño español"),
        
        ("Äußerst schön und größer", 
         "äußerst schön und größer"),
        
        ("Je n'ai cependant pas vraiment l'envie", 
         "je n'ai cependant pas vraiment l'envie"),
    ]
    
    print("=== Testing text normalization ===")
    for original, expected in test_cases:
        result = normalize_text(original, lower=True)
        status = "✓" if expected in result else "✗"
        print(f"{status} Input:    {original}")
        print(f"  Output:   {result}")
        print(f"  Expected: {expected}")
        print()


if __name__ == "__main__":
    validate_cleaning()