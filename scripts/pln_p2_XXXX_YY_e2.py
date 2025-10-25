"""
Script for loading a processed BoardGameGeek corpus, extracting features, 
and vectorizing reviews with TF-IDF and sentiment features.
"""

import os
from collections import Counter
from scipy.sparse import save_npz
import joblib

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.features.vectorization import ReviewVectorizer
from src.bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, VECTORS_DIR

# ----------------------------
# 1. Load processed corpus
# ----------------------------
corpus_path = os.path.join(CORPORA_DIR, "bgg_corpus.json")
LOGGER.info(f"Loading corpus from {corpus_path}")
corpus = Corpus.from_json(corpus_path)

# ----------------------------
# 2. Prepare input features
# ----------------------------
tokens_per_doc = []
langs = []
opinion_features = []
skipped_docs = 0

for i, doc in enumerate(corpus.documents):
    tokens_no_stop = doc.processed.get("tokens_no_stopwords")
    
    if not tokens_no_stop:
        skipped_docs += 1
        continue

    tokens_per_doc.append(tokens_no_stop)

    lang = doc.language
    langs.append(lang)

    vader = doc.linguistic_features.get("vader_scores", {})
    sentiment_words = doc.linguistic_features.get("sentiment_words", {})
    features = {**vader, **sentiment_words}
    opinion_features.append(features)

# ----------------------------
# 3. Summarize corpus statistics
# ----------------------------
total_docs = len(corpus.documents)
processed_docs = len(tokens_per_doc)
LOGGER.info(f"Total documents in corpus: {total_docs}")
LOGGER.info(f"Documents processed: {processed_docs}, skipped: {skipped_docs}")

if tokens_per_doc:
    lang_counts = Counter(langs)
    doc_lengths = [len(t) for t in tokens_per_doc]
    LOGGER.info(f"Language distribution: {dict(lang_counts)}")
    LOGGER.info(
        "Tokens per document: min=%d, max=%d, mean=%.1f" %
        (min(doc_lengths), max(doc_lengths), sum(doc_lengths)/len(doc_lengths))
    )

# ----------------------------
# 4. Initialize vectorizer
# ----------------------------
vec = ReviewVectorizer(max_features=8000, ngram_range=(1, 2), stopwords=None)

# ----------------------------
# 5. Fit-transform and save
# ----------------------------
if tokens_per_doc:
    LOGGER.info("Vectorizing reviews...")
    X = vec.fit_transform(tokens_per_doc, langs, opinion_features)
    
    os.makedirs(VECTORS_DIR, exist_ok=True)
    save_npz(os.path.join(VECTORS_DIR, "bgg_combined_matrix.npz"), X)
    joblib.dump(vec, os.path.join(VECTORS_DIR, "bgg_vectorizer.pkl"))
    
    LOGGER.info("✅ TF-IDF + sentiment vectorization complete")
else:
    LOGGER.error("❌ No documents with tokens found, vectorization skipped")