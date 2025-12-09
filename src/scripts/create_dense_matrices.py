"""
Module: create_static_embeddings.py
----------------------------------
This module provides utilities to generate document-level embeddings using 
pretrained static word embeddings (Word2Vec, GloVe, FastText) downloaded 
directly via the official gensim API.

Workflow:
1. Download/load pretrained embeddings via gensim.downloader.
2. Load processed corpus (tokenized, stopwords removed).
3. Compute document embeddings (average or sum).
4. Save the final embedding matrix, labels and metadata.

Usage example:
    python scripts/create_dense_matrices.py \
        --corpus path/to/bgg_corpus.json \
        --embedding_type word2vec \
        --agg avg
"""

import os
import argparse
import numpy as np
import joblib

from src.bgg_corpus.models import Corpus
from src.bgg_corpus.config import (
    EMBEDDINGS_DIR, 
    CORPORA_DIR, 
    CORPUS_NAME,
    VECTORS_DIR_EMB_W2V,
    VECTORS_DIR_EMB_GLOVE,
    VECTORS_DIR_EMB_FASTTEXT
)
os.environ["GENSIM_DATA_DIR"] = EMBEDDINGS_DIR # Where to store gensim models. By default, it uses ~/.gensim
import gensim.downloader as api
from src.bgg_corpus.resources import LOGGER


# -------------------------------------------------
# Mapping: embedding type → gensim model name
# -------------------------------------------------
GENSIM_MODELS = {
    "word2vec": "word2vec-google-news-300",
    "glove": "glove-wiki-gigaword-300",
    "fasttext": "fasttext-wiki-news-subwords-300"
}

# -------------------------------------------------
# Mapping: embedding type → where to save dense matrices
# -------------------------------------------------
OUTPUT_DIR_BY_EMB = {
    "word2vec": VECTORS_DIR_EMB_W2V,
    "glove": VECTORS_DIR_EMB_GLOVE,
    "fasttext": VECTORS_DIR_EMB_FASTTEXT
}


# -------------------------------------------------
# Load embeddings using gensim API
# -------------------------------------------------
def load_embeddings(emb_type: str):
    """
    Load pretrained embeddings using gensim.downloader, forcing models
    to be stored inside EMBEDDINGS_DIR.

    Returns
    -------
    KeyedVectors
    """
    os.environ["GENSIM_DATA_DIR"] = EMBEDDINGS_DIR  # override download path

    model_name = GENSIM_MODELS[emb_type]
    LOGGER.info(f"Loading gensim model '{model_name}' into {EMBEDDINGS_DIR} ...")

    kv = api.load(model_name)
    LOGGER.info(f"Loaded embeddings: vector size = {kv.vector_size}") # 300 for all three models
    return kv


# -------------------------------------------------
# Compute document vector
# -------------------------------------------------
def embed_document(tokens, kv, agg="avg"):
    """
    Aggregate word embeddings for a document.

    Parameters
    ----------
    tokens : list[str]
        Tokenized and cleaned words.
    kv : gensim KeyedVectors
    agg : str
        "avg" or "sum"
    """
    vectors = [kv[t] for t in tokens if t in kv]

    if not vectors:
        return np.zeros(kv.vector_size)

    mat = np.vstack(vectors)
    return np.mean(mat, axis=0) if agg == "avg" else np.sum(mat, axis=0)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate document embeddings using gensim static embeddings"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default=os.path.join(CORPORA_DIR, f"{CORPUS_NAME}.json"),
        help="Path to the processed corpus JSON"
    )
    parser.add_argument(
        "--embedding_type",
        choices=["word2vec", "glove", "fasttext"],
        default="glove"
    )
    parser.add_argument(
        "--agg",
        choices=["avg", "sum"],
        default="avg"
    )

    args = parser.parse_args()

    # Load embeddings from gensim
    kv = load_embeddings(args.embedding_type)

    # Load corpus
    LOGGER.info(f"Loading corpus from {args.corpus}")
    corpus = Corpus.from_json(args.corpus)

    tokens_per_doc = []
    categories = []

    for doc in corpus.documents:
        toks = doc.processed.get("tokens_no_stopwords")
        if not toks:
            continue
        tokens_per_doc.append(toks)
        categories.append(doc.category)

    # Compute document matrix
    LOGGER.info("Computing document embeddings ...")
    doc_matrix = np.zeros((len(tokens_per_doc), kv.vector_size))

    for i, toks in enumerate(tokens_per_doc):
        doc_matrix[i] = embed_document(toks, kv, agg=args.agg)

    # Output directory chosen according to embedding type
    output_dir = OUTPUT_DIR_BY_EMB[args.embedding_type]
    os.makedirs(output_dir, exist_ok=True)

    # Save outputs
    np.save(os.path.join(output_dir, "doc_embeddings.npy"), doc_matrix)
    np.save(os.path.join(output_dir, "labels.npy"), np.array(categories))
    joblib.dump(
        {"tokens_per_doc": tokens_per_doc, "agg": args.agg},
        os.path.join(output_dir, "metadata.pkl")
    )

    LOGGER.info(f"Saved embeddings to {output_dir}")


if __name__ == "__main__":
    main()