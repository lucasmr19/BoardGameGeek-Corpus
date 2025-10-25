from nltk.stem import WordNetLemmatizer
from .cleaning import normalize_text, extract_special_patterns
from .language import detect_language, get_spacy_lang_code, get_nltk_language, load_spacy_model_for, analyze_text_spacy
from .tokenization import sentence_segmentation, tokenize_and_filter, apply_stemming
from ..features import LinguisticFeaturesExtractor


from ..resources import STOPWORDS_CACHE

def process_review_item(item, lower=True, remove_stopwords=True, lemmatize=True,
                        correct_spelling=False, stemming=True):
    """
    High-level preprocessing function for review items.
    Includes text normalization, language detection, tokenization, stemming, lemmatization and feature extraction.
    """
    raw = item.get("raw_text")

    
    if not raw or not raw.strip():
        return {
            "username": item.get("username"),
            "rating": item.get("rating"),
            "timestamp": item.get("timestamp"),
            "raw_text": raw,
            "clean_text": "",
            "language_detected": "unknown",
            "language": "en",
            "sentences": [],
            "tokens": [],
            "tokens_no_stopwords": [],
            "stems": [],
            "lemmas": [],
            "pos_tags": [],
            "dependencies": [],
            "num_tokens": 0,
            "num_sentences": 0,
            "patterns": {
                "emails": [], "dates": [], "phones": [], "hashtags": [],
                "mentions": [], "urls": [], "emojis": []
            }
        }
    
    # 1. Special pattern extraction and text normalization
    special_patterns = extract_special_patterns(raw)
    clean = normalize_text(raw, lower=lower, correct_spelling=correct_spelling)
    
    # 2. Language detection
    detected_lang = detect_language(clean if clean else raw)
    spacy_lang = get_spacy_lang_code(detected_lang)
    nltk_lang = get_nltk_language(spacy_lang)

    # 3. Sentence segmentation and tokenization
    sentences = sentence_segmentation(clean, nltk_lang)
    stop_words_set = STOPWORDS_CACHE.get(nltk_lang, set())
    tokens, tokens_no_stop = tokenize_and_filter(clean, nltk_lang, stop_words_set, remove_stopwords)
        
    # 4. SpaCy analysis for POS tagging and dependencies
    spacy_analysis = analyze_text_spacy(clean, detected_lang)
    
    # 5. Stemming
    stems = apply_stemming(tokens_no_stop, spacy_lang) if stemming else []
    
    # 6. Lemmas
    lemmas_final = []
    if lemmatize:
        if spacy_analysis.get("spacy_used") and spacy_analysis.get("pos_tags"):
            # Map tokens_no_stop to lemmas in spaCy
            nlp = load_spacy_model_for(spacy_lang)
            if nlp:
                lemmas_final = []
                for token_text in tokens_no_stop:
                    doc_token = nlp(token_text)
                    lemma = doc_token[0].lemma_ if doc_token else token_text
                    lemmas_final.append(lemma)
        else:
            # Fallback: lemmatization with NLTK for each token_no_stop
            lemmatizer = WordNetLemmatizer()
            lemmas_final = [lemmatizer.lemmatize(t) for t in tokens_no_stop]
    
    dependencies = spacy_analysis.get("dependencies", [])

    # 7. Linguistic feature extraction
    extractor = LinguisticFeaturesExtractor()
    linguistic_features = extractor.extract_features(
        lemmas=lemmas_final,
        tokens_no_stopwords=tokens_no_stop,
        dependencies=dependencies,
        sentences=sentences,
        raw_text=raw
    )
    
    # 8. Output assembly
    out = {
        "username": item.get("username"),
        "rating": item.get("rating"),
        "timestamp": item.get("timestamp"),
        "raw_text": raw,
        "clean_text": clean,
        "language_detected": detected_lang,
        "language": spacy_lang,
        "sentences": sentences,
        "tokens": tokens,
        "tokens_no_stopwords": tokens_no_stop,
        "stems": stems,
        "lemmas": lemmas_final,
        "pos_tags": spacy_analysis.get("pos_tags", []),
        "dependencies": dependencies,
        "linguistic_features": linguistic_features,
        "patterns": special_patterns
    }

    return out