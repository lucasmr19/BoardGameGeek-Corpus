"""
Vectorization module for text and opinion features with multilingual awareness
"""

from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import hstack


class ReviewVectorizer:
    """Generate textual and opinion-based vector representations, keeping track of language."""

    def __init__(self, max_features: int = 5000, ngram_range: tuple[float, float] = (1, 2), stopwords: set = None):
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=stopwords
        )
        self.dict_vectorizer = DictVectorizer(sparse=True)

    @staticmethod
    def _prefix_tokens_with_language(tokens_per_doc: List[List[str]], langs: List[str]) -> List[str]:
        """Add a language prefix to each token to make TF-IDF multilingual-aware."""
        prefixed_texts = []
        for tokens, lang in zip(tokens_per_doc, langs):
            # Evitamos conflictos de tokens entre idiomas
            prefixed_tokens = [f"{lang}_{tok}" for tok in tokens]
            prefixed_texts.append(" ".join(prefixed_tokens))
        return prefixed_texts

    def fit_transform(
        self,
        tokens_per_doc: List[List[str]],
        langs: List[str],
        opinion_features: List[Dict[str, Any]]
    ):
        """Fit both TF-IDF and opinion-based vectorizers, and combine."""
        texts_prefixed = self._prefix_tokens_with_language(tokens_per_doc, langs)
        X_tfidf = self.tfidf.fit_transform(texts_prefixed)
        X_dict = self.dict_vectorizer.fit_transform(opinion_features)
        return hstack([X_tfidf, X_dict])

    def transform(
        self,
        tokens_per_doc: List[List[str]],
        langs: List[str],
        opinion_features: List[Dict[str, Any]]
    ):
        """Transform new data using fitted vectorizers."""
        texts_prefixed = self._prefix_tokens_with_language(tokens_per_doc, langs)
        X_tfidf = self.tfidf.transform(texts_prefixed)
        X_dict = self.dict_vectorizer.transform(opinion_features)
        return hstack([X_tfidf, X_dict])
