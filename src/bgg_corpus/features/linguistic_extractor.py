"""
Linguistic Features Extraction Module
"""

from typing import Dict, List, Any
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from .lexicons import SentimentLexicon


class LinguisticFeaturesExtractor:
    """Extract linguistic and opinion-related features from board game reviews."""

    def __init__(self):
        self.lexicon = SentimentLexicon()
        self.sia = SentimentIntensityAnalyzer()

    # ======================================================
    # ============ MAIN EXTRACTION PIPELINE ================
    # ======================================================
    def extract_features(
        self,
        lemmas: List[str],
        tokens_no_stopwords: List[str],
        dependencies: List[str],
        sentences: List[str],
        raw_text: str = ""
    ) -> Dict[str, Any]:
        """Extract linguistic features given processed text components."""
        features = self._empty_features()

        # 1️⃣ --- Sentiment Lexicon Features ---
        pos_count = sum(1 for w in tokens_no_stopwords if w in self.lexicon.positive_words)
        neg_count = sum(1 for w in tokens_no_stopwords if w in self.lexicon.negative_words)
        total = pos_count + neg_count
        ratio = pos_count / max(total, 1)
        features["sentiment_words"].update({
            "positive_count": pos_count,
            "negative_count": neg_count,
            "total_count": total,
            "ratio": ratio
        })

        # 2️⃣ --- Negation Detection ---
        neg_positions = [i for i, w in enumerate(tokens_no_stopwords) if w in self.lexicon.negation_words]
        neg_scope = self._negation_scope(tokens_no_stopwords, neg_positions, window=3)
        features["negations"].update({
            "count": len(neg_positions),
            "positions": neg_positions,
            "negated_terms": neg_scope
        })

        # 3️⃣ --- Intensifiers and Mitigators ---
        intensifiers = [w for w in tokens_no_stopwords if w in self.lexicon.intensifiers]
        mitigators = [w for w in tokens_no_stopwords if w in self.lexicon.mitigators]
        features["intensifiers_mitigators"].update({
            "intensifiers": len(intensifiers),
            "mitigators": len(mitigators),
            "total_modifiers": len(intensifiers) + len(mitigators),
            "examples": {"intensifiers": intensifiers, "mitigators": mitigators}
        })

        # 4️⃣ --- Domain-Specific Terms + Nearby Sentiments ---
        domain_features = {}
        for cat, terms in self.lexicon.domain_terms.items():
            mentions = [w for w in lemmas if w in terms]
            nearby_sent = [
                w for w in tokens_no_stopwords
                if any(abs(i - j) <= 3
                       for i, w2 in enumerate(tokens_no_stopwords)
                       for j, t in enumerate(mentions) if w2 == t)
                and w in self.lexicon.positive_words.union(self.lexicon.negative_words)
            ]
            domain_features[cat] = {
                "count": len(mentions),
                "mentions": mentions,
                "nearby_sentiments": nearby_sent
            }
        features["domain_specific"] = domain_features

        # 5️⃣ --- Vader Sentiment Scores ---
        features["vader_scores"] = self._extract_vader_scores(raw_text)

        # 6️⃣ --- Generic Linguistic Features ---
        features["syntactic_complexity"] = self._syntactic_features(tokens_no_stopwords, dependencies)

        # 7️⃣ --- Sentence-Level Sentiment Dynamics ---
        features["sentence_level"] = self._extract_sentence_level_features(sentences)

        return features

    # ======================================================
    # ============ HELPER FUNCTIONS ========================
    # ======================================================
    def _negation_scope(self, tokens: List[str], neg_positions: List[int], window: int = 3) -> List[str]:
        """Return words within a small window after negation markers."""
        negated_terms = []
        for pos in neg_positions:
            scope = tokens[pos + 1: pos + window + 1]
            negated_terms.extend(scope)
        return negated_terms

    def _extract_vader_scores(self, text: str) -> Dict[str, float]:
        """Extract VADER sentiment scores."""
        scores = self.sia.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg']
        }

    def _syntactic_features(self, tokens: List[str], dependencies: List[str]) -> Dict[str, Any]:
        """Compute basic linguistic complexity and size metrics."""
        avg_token_length = np.mean([len(t) for t in tokens]) if tokens else 0
        dep_depths = [len(dep.split("/")) for dep in dependencies if isinstance(dep, str)]
        avg_dep = sum(dep_depths) / len(dep_depths) if dep_depths else 0

        return {
            "num_tokens_no_stop": len(tokens),
            "avg_token_no_stop_length": avg_token_length,
            "avg_dep_depth": avg_dep,
            "num_dependencies": len(dep_depths)
        }

    def _extract_sentence_level_features(self, sentences: List[str]) -> Dict[str, Any]:
        """Extract sentence-level sentiment features."""
        num_sentences = len(sentences)
        sentence_sentiments = [self.sia.polarity_scores(s)["compound"] for s in sentences]
        avg_sentiment = np.mean(sentence_sentiments) if sentence_sentiments else 0
        variance = np.var(sentence_sentiments) if len(sentence_sentiments) > 1 else 0
        return {
            'num_sentences': num_sentences,
            'avg_sentiment': avg_sentiment,
            'sentiment_variance': variance,
            'sentence_sentiments': sentence_sentiments
        }

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty structure for all feature categories."""
        return {
            "sentiment_words": {},
            "negations": {},
            "intensifiers_mitigators": {},
            "domain_specific": {},
            "vader_scores": {},
            "syntactic_complexity": {},
            "sentence_level": {}
        }