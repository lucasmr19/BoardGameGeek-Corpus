"""
Linguistic Features Extraction Module
"""

from typing import Dict, List, Any
from collections import Counter
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from .lexicons import SentimentLexicon


class LinguisticFeaturesExtractor:
    """Extract linguistic features and return a single dict ready for DictVectorizer,
       now including hedge-related features.
    """

    def __init__(self):
        self.lexicon = SentimentLexicon()
        self.sia = SentimentIntensityAnalyzer()

    def extract_features(
        self,
        lemmas: List[str],
        tokens_no_stopwords: List[str],
        dependencies: List[str],
        sentences: List[str],
        raw_text: str = ""
    ) -> Dict[str, Any]:
        """
        Return ONE dict with:
          - numeric features (floats/ints)
          - sequence-of-strings features (lists of strings) for counting by DictVectorizer
        """
        # Normalizar
        lemmas = [w.lower() for w in lemmas]
        tokens = [w.lower() for w in tokens_no_stopwords]

        out: Dict[str, Any] = {}

        # ---------- Basic sentiment and numeric features ----------
        pos_words = [w for w in tokens if w in self.lexicon.positive_words]
        neg_words = [w for w in tokens if w in self.lexicon.negative_words]
        out["sentiment.pos_count"] = len(pos_words)
        out["sentiment.neg_count"] = len(neg_words)
        out["sentiment.total"] = len(pos_words) + len(neg_words)
        out["sentiment.pos_ratio"] = (len(pos_words) / max(out["sentiment.total"], 1))

        # Vader numeric
        vader = self._extract_vader_scores(raw_text)
        out["vader.compound"] = vader["compound"]
        out["vader.pos"] = vader["pos"]
        out["vader.neu"] = vader["neu"]
        out["vader.neg"] = vader["neg"]

        # Syntactic numeric
        synt = self._syntactic_features(tokens, dependencies)
        out["syntactic.num_tokens_no_stop"] = synt["num_tokens_no_stop"]
        out["syntactic.avg_token_no_stop_length"] = synt["avg_token_no_stop_length"]
        out["syntactic.avg_dep_depth"] = synt["avg_dep_depth"]
        out["syntactic.num_dependencies"] = synt["num_dependencies"]

        # Sentence-level numeric
        sent_level = self._extract_sentence_level_features(sentences)
        out["sentence.num_sentences"] = sent_level["num_sentences"]
        out["sentence.avg_sentiment"] = sent_level["avg_sentiment"]
        out["sentence.sentiment_variance"] = sent_level["sentiment_variance"]

        # ---------- Hedging features (numeric) ----------
        # counts per hedge type
        hedge_positions = [i for i, w in enumerate(tokens) if w in self.lexicon.hedge_words]
        propositional_positions = [i for i, w in enumerate(tokens) if w in self.lexicon.propositional_hedges]
        relational_positions = [i for i, w in enumerate(tokens) if w in self.lexicon.relational_hedges]
        discourse_positions = [i for i, w in enumerate(tokens) if w in self.lexicon.discourse_markers]
        all_hedge_positions = sorted(set(hedge_positions + propositional_positions +
                                         relational_positions + discourse_positions))

        out["hedge.count"] = len(all_hedge_positions)
        out["hedge.count_type.hedge_words"] = len(hedge_positions)
        out["hedge.count_type.propositional"] = len(propositional_positions)
        out["hedge.count_type.relational"] = len(relational_positions)
        out["hedge.count_type.discourse_markers"] = len(discourse_positions)

        # density: hedges per token
        out["hedge.density"] = out["hedge.count"] / max(len(tokens), 1)

        # Hedge-to-sentiment relations: how many sentiment words appear within window of any hedge
        # (helps detect softened sentiment)
        window = 3
        nearby_sentiment_from_hedges = self._count_sentiment_near_positions(tokens, all_hedge_positions, window)
        out["hedge.nearby_sentiment_count"] = nearby_sentiment_from_hedges
        out["hedge.nearby_sentiment_ratio"] = nearby_sentiment_from_hedges / max(out["sentiment.total"], 1)

        # If you want: proportion of hedges that are propositional / relational / discourse
        out["hedge.prop_propositional"] = (out["hedge.count_type.propositional"] / max(out["hedge.count"], 1))
        out["hedge.prop_relational"] = (out["hedge.count_type.relational"] / max(out["hedge.count"], 1))
        out["hedge.prop_discourse"] = (out["hedge.count_type.discourse_markers"] / max(out["hedge.count"], 1))

        # ---------- Sequence-of-strings features (for DictVectorizer) ----------
        # Keep duplicates so DictVectorizer counts occurrences

        # 1) Hedge tokens per subtype
        self._add_counts("hedge.words", out, [tokens[i] for i in hedge_positions])
        self._add_counts("hedge.propositional", out, [tokens[i] for i in propositional_positions])
        self._add_counts("hedge.relational", out, [tokens[i] for i in relational_positions])
        self._add_counts("hedge.discourse_marker", out, [tokens[i] for i in discourse_positions])
        self._add_counts("hedge.all", out, [tokens[i] for i in all_hedge_positions])

        # 2) Sentiment words near hedges as string list (useful to detect softened sentiment)
        nearby_words = self._vocab_near_positions(tokens, all_hedge_positions, window,
                                                                         vocab=self.lexicon.positive_words | self.lexicon.negative_words)

        self._add_counts("hedge.nearby_sentiment_words", out, nearby_words)
        
        # 3) Domain-specific (keep as before)
        for cat, terms in self.lexicon.domain_terms.items():
            terms_set = set(terms)
            matches = [w for w in lemmas if w in terms_set]
            self._add_counts(f"domain.{cat}", out, matches)

        # 4) Intensifiers / mitigators / negations (lists with duplicates)
        self._add_counts("lexicon.intensifier", out, [w for w in tokens if w in self.lexicon.intensifiers])
        self._add_counts("lexicon.mitigator", out, [w for w in tokens if w in self.lexicon.mitigators])
        self._add_counts("lexicon.negation", out, [w for w in tokens if w in self.lexicon.negation_words])

        # 5) Positive / negative words (for counting by DictVectorizer)
        self._add_counts("lexicon.pos_word", out, pos_words)
        self._add_counts("lexicon.neg_word", out, neg_words)

        # 6) Negation scope words (list)
        neg_positions = [i for i, w in enumerate(tokens) if w in self.lexicon.negation_words]
        scope_words= self._negation_scope(tokens, neg_positions, window=3)
        self._add_counts("negation.scope_word", out, scope_words)

        # 7) Optionally: raw tokens (comment if you don't want full BoW)
        #out["raw.tokens"] = tokens[:]   # WARNING: may create huge vocabulary

        return out

    # ---------------- helpers ----------------
    def _add_counts(self, prefix: str, out: Dict[str, Any], words: list):
        """Add counts {prefix=value: count} to output."""
        for w, c in Counter(words).items():
            out[f"{prefix}={w}"] = c
        return out

    def _negation_scope(self, tokens: List[str], neg_positions: List[int], window: int = 3) -> List[str]:
        negated_terms = []
        for pos in neg_positions:
            scope = tokens[pos + 1: pos + window + 1]
            negated_terms.extend(scope)
        return negated_terms

    def _count_sentiment_near_positions(self, tokens: List[str], positions: List[int], window: int = 3) -> int:
        """Return the number of sentiment-word occurrences within +/-window of any position in positions."""
        if not positions:
            return 0
        sent_vocab = set(self.lexicon.positive_words) | set(self.lexicon.negative_words)
        count = 0
        for pos in positions:
            start = max(0, pos - window)
            end = min(len(tokens), pos + window + 1)
            for w in tokens[start:end]:
                if w in sent_vocab:
                    count += 1
        return count

    def _vocab_near_positions(self, tokens: List[str], positions: List[int], window: int, vocab: set) -> List[str]:
        """Return list of vocab words (with duplicates) that appear within +/-window of any position."""
        found = []
        for pos in positions:
            start = max(0, pos - window)
            end = min(len(tokens), pos + window + 1)
            for w in tokens[start:end]:
                if w in vocab:
                    found.append(w)
        return found

    def _extract_vader_scores(self, text: str) -> Dict[str, float]:
        scores = self.sia.polarity_scores(text)
        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neu': scores['neu'],
            'neg': scores['neg']
        }

    def _syntactic_features(self, tokens: List[str], dependencies: List[str]) -> Dict[str, Any]:
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
        num_sentences = len(sentences)
        sentence_sentiments = [self.sia.polarity_scores(s)["compound"] for s in sentences]
        avg_sentiment = np.mean(sentence_sentiments) if sentence_sentiments else 0
        variance = np.var(sentence_sentiments) if len(sentence_sentiments) > 1 else 0
        return {
            'num_sentences': num_sentences,
            'avg_sentiment': avg_sentiment,
            'sentiment_variance': variance,
        }
