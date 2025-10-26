"""Module for text analysis using spaCy."""
from .language import get_spacy_lang_code, load_spacy_model_for

def analyze_text_spacy(text, detected_code):
    """
    Analyze the given text using spaCy.

    Returns:
    - sentences: List of sentences in the text.
    - tokens: List of tokens in the text.
    - lemmas: List of lemmas for the tokens.
    - pos_tags: List of tuples (token, POS, tag).
    - dependencies: List of tuples (token, dep, head).
    - entities: List of tuples (ent.text, ent.label_).
    
    """
    code = get_spacy_lang_code(detected_code)
    nlp = load_spacy_model_for(code)

    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    tokens = [t.text for t in doc if not t.is_space]
    lemmas = [t.lemma_ for t in doc if not t.is_space]
    pos_tags = [(t.text, t.pos_, t.tag_) for t in doc]
    dependencies = [(t.text, t.dep_, t.head.text) for t in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    
    return sentences, tokens, lemmas, pos_tags, dependencies, entities