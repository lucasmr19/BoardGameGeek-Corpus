"""Module for text analysis using spaCy."""

import re
from .language import get_spacy_lang_code, load_spacy_model_for

def analyze_text_spacy(text, detected_code, stop_words_set=None, remove_stopwords=True):
    """
    Analyze the given text using spaCy, using consistent NLTK-based stopword filtering.

    Args:
        text (str): Input text.
        detected_code (str): Detected language code (e.g. 'en', 'es').
        stop_words_set (set[str], optional): Stopwords for the detected language (from NLTK or project cache).
        remove_stopwords (bool): Whether to remove stopwords from the token list.

    Returns:
        sentences (list[str])
        tokens (list[str])
        tokens_no_stopwords (list[str])
        lemmas (list[str])
        pos_tags (list[tuple[str, str, str]])
        dependencies (list[tuple[str, str, str]])
        entities (list[tuple[str, str]])
    """

    code = get_spacy_lang_code(detected_code)
    nlp = load_spacy_model_for(code)
    doc = nlp(text)

    # --- Sentences ---
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # --- Tokens (no spaces or punctuation) ---
    tokens = [
        t.text.lower()
        for t in doc
        if not (t.is_space or t.is_punct)
    ]

    # --- Tokens no stopwords (based on NLTK/project stopword set) ---
    tokens_no_stopwords = [
        t.text.lower()
        for t in doc
        if not (t.is_space or t.is_punct)
        and (not remove_stopwords or t.text.lower() not in (stop_words_set or set()))
        and re.search(r"\w", t.text)
    ]

    # --- Lemmas ---
    lemmas = [
        t.lemma_.lower().strip()
        for t in doc
        if not (t.is_space or t.is_punct)
    ]

    # --- POS tags ---
    pos_tags = [
        (t.text, t.pos_, t.tag_)
        for t in doc
        if not t.is_space
    ]

    # --- Dependencies ---
    dependencies = [
        (t.text, t.dep_, t.head.text)
        for t in doc
        if not t.is_space
    ]

    # --- Named entities ---
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return sentences, tokens, tokens_no_stopwords, lemmas, pos_tags, dependencies, entities
