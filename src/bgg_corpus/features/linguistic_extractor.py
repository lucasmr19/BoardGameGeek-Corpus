"""
Linguistic Features Extraction Module

This module extracts comprehensive linguistic features for sentiment analysis,
including sentiment, syntactic complexity, hedging patterns, subjectivity markers,
and compositional effects (negation/intensification on sentiment).
"""

from typing import Dict, List, Any, Tuple
from collections import Counter
import numpy as np
import re
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import ngrams
import textstat

from .lexicons import SentimentLexicon


class LinguisticFeaturesExtractor:
    """
    Extract linguistic features and return a single dict ready for DictVectorizer.
    
    This extractor computes a comprehensive set of features organized into categories:
    
    1. **Sentiment Features**: Basic sentiment counts, VADER scores, sentence-level sentiment
    2. **Lexical Features**: Type-token ratio, hapax legomena, entropy
    3. **Syntactic Features**: Token counts, dependency depth, sentence structure
    4. **Hedging Features**: Hedge words, propositional/relational hedges, discourse markers
    5. **Negation Features**: Negation scope, negation-sentiment interactions
    6. **Compositional Features**: Sentiment polarity adjusted by negation and intensifiers
    7. **Subjectivity Markers**: Explicit subjectivity indicators ("I think", "probably", etc.)
    8. **Readability Features**: Flesch-Kincaid grade, reading ease, complexity
    9. **Domain Features**: Domain-specific term occurrences
    10. **POS Features**: Part-of-speech distribution ratios
    
    Features are returned as:
    - Numeric features (floats/ints) for direct use
    - Sequence-of-strings features (lists) for counting by DictVectorizer
    """

    def __init__(self):
        """Initialize the extractor with sentiment lexicon and VADER analyzer."""
        self.lexicon = SentimentLexicon()
        self.sia = SentimentIntensityAnalyzer()
        
        # Subjectivity markers for explicit opinion/uncertainty
        self.subjectivity_markers = {
            'i_think', 'i_believe', 'i_feel', 'in_my_opinion', 'personally',
            'it_seems', 'it_appears', 'probably', 'possibly', 'perhaps',
            'maybe', 'supposedly', 'arguably', 'apparently', 'presumably'
        }

    def extract_features(
        self,
        lemmas: List[str],
        tokens_no_stopwords: List[str],
        dependencies: List[str],
        sentences: List[str],
        pos_tags: List[tuple],
        raw_text: str = ""
    ) -> Dict[str, Any]:
        """
        Extract all linguistic features from preprocessed text.
        
        Args:
            lemmas: List of lemmatized tokens
            tokens_no_stopwords: List of tokens with stopwords removed
            dependencies: List of dependency parse strings
            sentences: List of sentence strings
            pos_tags: List of (token, pos, tag) tuples
            raw_text: Original raw text string
            
        Returns:
            Dictionary with numeric features (float/int) and sequence features (lists)
            ready for use with sklearn's DictVectorizer.
        """
        out: Dict[str, Any] = {}

        # ========== 1. BASIC SENTIMENT FEATURES ==========
        """
        Basic lexicon-based sentiment counting using positive/negative word lists.
        Provides raw counts and ratios of sentiment-bearing words.
        """
        pos_words = [w for w in tokens_no_stopwords if w in self.lexicon.positive_words]
        neg_words = [w for w in tokens_no_stopwords if w in self.lexicon.negative_words]
        out["sentiment.pos_count"] = len(pos_words)
        out["sentiment.neg_count"] = len(neg_words)
        out["sentiment.total"] = len(pos_words) + len(neg_words)
        out["sentiment.pos_ratio"] = (len(pos_words) / max(out["sentiment.total"], 1))

        # ========== 2. VADER SENTIMENT SCORES ==========
        """
        VADER (Valence Aware Dictionary and sEntiment Reasoner) provides
        rule-based sentiment with consideration of negation, intensifiers, etc.
        """
        vader = self._extract_vader_scores(raw_text)
        out["vader.compound"] = vader["compound"]
        out["vader.pos"] = vader["pos"]
        out["vader.neu"] = vader["neu"]
        out["vader.neg"] = vader["neg"]

        # ========== 3. SYNTACTIC FEATURES ==========
        """
        Measures of syntactic complexity including token counts, lengths,
        and dependency parse depth (deeper trees = more complex syntax).
        """
        synt = self._syntactic_features(tokens_no_stopwords, dependencies)
        out["syntactic.num_tokens_no_stop"] = synt["num_tokens_no_stop"]
        out["syntactic.avg_token_no_stop_length"] = synt["avg_token_no_stop_length"]
        out["syntactic.avg_dep_depth"] = synt["avg_dep_depth"]
        out["syntactic.num_dependencies"] = synt["num_dependencies"]
        
        # ========== 4. LEXICAL DIVERSITY FEATURES ==========
        """
        Measures of vocabulary richness and distribution:
        - TTR (Type-Token Ratio): unique words / total words
        - Hapax ratio: words appearing once / total words
        - Entropy: information-theoretic measure of word distribution
        """
        word_counts = Counter(tokens_no_stopwords)
        out["lexical.ttr"] = len(word_counts) / max(len(tokens_no_stopwords), 1)
        out["lexical.hapax_ratio"] = sum(1 for c in word_counts.values() if c == 1) / max(len(tokens_no_stopwords), 1)
        freqs = np.array(list(word_counts.values())) / max(sum(word_counts.values()), 1)
        out["lexical.entropy"] = -(freqs * np.log2(freqs + 1e-10)).sum()

        # ========== 5. SENTENCE-LEVEL FEATURES ==========
        """
        Aggregated statistics across sentences including sentiment
        variance (how much sentiment fluctuates between sentences).
        """
        sent_level = self._extract_sentence_level_features(sentences)
        out["sentence.num_sentences"] = sent_level["num_sentences"]
        out["sentence.avg_sentiment"] = sent_level["avg_sentiment"]
        out["sentence.sentiment_variance"] = sent_level["sentiment_variance"]
        
        # ========== 6. READABILITY & COMPLEXITY ==========
        """
        Standard readability metrics and syllable-based complexity measures.
        Higher FK grade = more complex text; higher ease = easier to read.
        """
        out["readability.fk_grade"] = textstat.flesch_kincaid_grade(raw_text)
        out["readability.ease"] = textstat.flesch_reading_ease(raw_text)
        out["readability.complex_word_ratio"] = textstat.polysyllabcount(raw_text) / max(len(tokens_no_stopwords), 1)

        # ========== 7. HEDGING FEATURES ==========
        """
        Hedges indicate uncertainty or softened claims:
        - Hedge words: general hedging (e.g., "possibly", "somewhat")
        - Propositional: epistemic uncertainty (e.g., "might", "could")
        - Relational: attribution/distancing (e.g., "according to")
        - Discourse markers: discourse-level hedging (e.g., "however", "although")
        
        Includes density and proximity to sentiment words (softened sentiment).
        """
        hedge_positions = [i for i, w in enumerate(tokens_no_stopwords) if w in self.lexicon.hedge_words]
        propositional_positions = [i for i, w in enumerate(tokens_no_stopwords) if w in self.lexicon.propositional_hedges]
        relational_positions = [i for i, w in enumerate(tokens_no_stopwords) if w in self.lexicon.relational_hedges]
        discourse_positions = [i for i, w in enumerate(tokens_no_stopwords) if w in self.lexicon.discourse_markers]
        all_hedge_positions = sorted(set(hedge_positions + propositional_positions +
                                         relational_positions + discourse_positions))

        out["hedge.count"] = len(all_hedge_positions)
        out["hedge.count_type.hedge_words"] = len(hedge_positions)
        out["hedge.count_type.propositional"] = len(propositional_positions)
        out["hedge.count_type.relational"] = len(relational_positions)
        out["hedge.count_type.discourse_markers"] = len(discourse_positions)
        out["hedge.density"] = out["hedge.count"] / max(len(tokens_no_stopwords), 1)

        # Hedge-to-sentiment relations
        window = 3
        nearby_sentiment_from_hedges = self._count_sentiment_near_positions(tokens_no_stopwords, all_hedge_positions, window)
        out["hedge.nearby_sentiment_count"] = nearby_sentiment_from_hedges
        out["hedge.nearby_sentiment_ratio"] = nearby_sentiment_from_hedges / max(out["sentiment.total"], 1)

        out["hedge.prop_propositional"] = (out["hedge.count_type.propositional"] / max(out["hedge.count"], 1))
        out["hedge.prop_relational"] = (out["hedge.count_type.relational"] / max(out["hedge.count"], 1))
        out["hedge.prop_discourse"] = (out["hedge.count_type.discourse_markers"] / max(out["hedge.count"], 1))

        # ========== 8. NEGATION FEATURES ==========
        """
        Captures negation and its scope (words following negation).
        Negation-sentiment ratio helps detect reversed polarity contexts.
        """
        neg_positions = [i for i, w in enumerate(tokens_no_stopwords) if w in self.lexicon.negation_words]
        scope_words = self._negation_scope(tokens_no_stopwords, neg_positions, window=3)
        
        neg_words_set = set(self.lexicon.negation_words)
        out["negation.count"] = sum(1 for t in tokens_no_stopwords if t in neg_words_set)
        out["negation.sentiment_ratio"] = out["negation.count"] / max(out["sentiment.total"], 1)

        # ========== 9. COMPOSITIONAL POLARITY ADJUSTMENT ==========
        """
        **NEW**: Adjusts sentiment polarity based on compositional effects:
        - Negation flips polarity (e.g., "not good" becomes negative)
        - Intensifiers amplify polarity (e.g., "very good" is more positive)
        
        This captures how modifiers change the effective sentiment strength
        beyond simple word counting.
        """
        compositional = self._compute_compositional_polarity(
            tokens_no_stopwords, neg_positions, pos_words, neg_words
        )
        out["compositional.adjusted_pos_count"] = compositional["adjusted_pos_count"]
        out["compositional.adjusted_neg_count"] = compositional["adjusted_neg_count"]
        out["compositional.adjusted_net_sentiment"] = compositional["adjusted_net_sentiment"]
        out["compositional.intensifier_boost"] = compositional["intensifier_boost"]
        out["compositional.negation_flips"] = compositional["negation_flips"]

        # ========== 10. SUBJECTIVITY MARKERS ==========
        """
        **NEW**: Explicit markers of subjectivity and personal opinion.
        These phrases signal that content is opinion-based rather than factual:
        - First-person opinion: "I think", "I believe", "I feel"
        - Epistemic markers: "probably", "possibly", "perhaps", "maybe"
        - Appearance markers: "it seems", "it appears", "apparently"
        
        Higher counts suggest more subjective, less certain content.
        """
        subjectivity = self._extract_subjectivity_markers(tokens_no_stopwords, raw_text)
        out["subjectivity.marker_count"] = subjectivity["marker_count"]
        out["subjectivity.marker_density"] = subjectivity["marker_density"]
        out["subjectivity.first_person_opinion_count"] = subjectivity["first_person_opinion_count"]
        out["subjectivity.epistemic_count"] = subjectivity["epistemic_count"]
        out["subjectivity.appearance_count"] = subjectivity["appearance_count"]

        # ========== 11. PUNCTUATION EMPHASIS ==========
        """
        Repeated punctuation and exclamations signal emotional intensity.
        """
        out["punct.exclamation_count"] = raw_text.count("!")
        out["punct.repeated_punct_count"] = sum(1 for m in re.findall(r"([!?.,])\1+", raw_text))
        
        # ========== 12. POSITIONAL SENTIMENT ==========
        """
        Sentiment at specific positions (first/last sentence, extremes)
        can be particularly influential in overall perception.
        """
        sentence_sentiments = [self.sia.polarity_scores(s)["compound"] for s in sentences]
        out["sentiment.first_sentence"] = sentence_sentiments[0] if sentence_sentiments else 0
        out["sentiment.last_sentence"] = sentence_sentiments[-1] if sentence_sentiments else 0
        out["sentiment.max_sentence"] = max(sentence_sentiments) if sentence_sentiments else 0
        out["sentiment.min_sentence"] = min(sentence_sentiments) if sentence_sentiments else 0
        
        # ========== 13. CO-OCCURRENCE FEATURES ==========
        """
        N-gram patterns capturing domain-sentiment associations.
        E.g., "terrible camera" vs "excellent camera" in tech reviews.
        """
        sentiment_vocab = set(self.lexicon.positive_words) | set(self.lexicon.negative_words)
        domain_terms = set([w for terms in self.lexicon.domain_terms.values() for w in terms])
        
        unigrams = tokens_no_stopwords
        bigrams = list(ngrams(tokens_no_stopwords, 2))
        trigrams = list(ngrams(tokens_no_stopwords, 3))
        out["sentiment_unigram_count"] = sum(1 for a in unigrams if a in sentiment_vocab)
        out["domain_sentiment_bigram_count"] = sum(1 for a, b in bigrams if a in domain_terms and b in sentiment_vocab)
        out["domain_sentiment_trigram_count"] = sum(1 for a, b, c in trigrams if a in domain_terms and b in sentiment_vocab and c in sentiment_vocab)

        # ========== 14. POS DISTRIBUTION ==========
        """
        Part-of-speech ratios can indicate writing style:
        - High adjective/adverb ratio: more descriptive, evaluative
        - High noun ratio: more concrete, factual
        - High verb ratio: more action-oriented
        """
        pos_counts = Counter([pos for _, pos, _ in pos_tags])
        total_pos = sum(pos_counts.values())
        out["pos.adj_ratio"] = pos_counts.get("ADJ", 0) / max(total_pos, 1)
        out["pos.adv_ratio"] = pos_counts.get("ADV", 0) / max(total_pos, 1)
        out["pos.noun_ratio"] = pos_counts.get("NOUN", 0) / max(total_pos, 1)
        out["pos.verb_ratio"] = pos_counts.get("VERB", 0) / max(total_pos, 1)

        # ========== 15. SEQUENCE FEATURES (for DictVectorizer) ==========
        """
        These list-based features allow DictVectorizer to count specific words/patterns.
        Each occurrence is preserved to capture frequency information.
        """
        # Hedge tokens by subtype
        self._add_counts("hedge.words", out, [tokens_no_stopwords[i] for i in hedge_positions])
        self._add_counts("hedge.propositional", out, [tokens_no_stopwords[i] for i in propositional_positions])
        self._add_counts("hedge.relational", out, [tokens_no_stopwords[i] for i in relational_positions])
        self._add_counts("hedge.discourse_marker", out, [tokens_no_stopwords[i] for i in discourse_positions])
        self._add_counts("hedge.all", out, [tokens_no_stopwords[i] for i in all_hedge_positions])

        # Sentiment words near hedges
        nearby_words = self._vocab_near_positions(
            tokens_no_stopwords, all_hedge_positions, window,
            vocab=self.lexicon.positive_words | self.lexicon.negative_words
        )
        self._add_counts("hedge.nearby_sentiment_words", out, nearby_words)
        
        # Domain-specific terms
        for cat, terms in self.lexicon.domain_terms.items():
            terms_set = set(terms)
            matches = [w for w in lemmas if w in terms_set]
            self._add_counts(f"domain.{cat}", out, matches)

        # Intensifiers, mitigators, negations
        self._add_counts("lexicon.intensifier", out, [w for w in tokens_no_stopwords if w in self.lexicon.intensifiers])
        self._add_counts("lexicon.mitigator", out, [w for w in tokens_no_stopwords if w in self.lexicon.mitigators])
        self._add_counts("lexicon.negation", out, [w for w in tokens_no_stopwords if w in self.lexicon.negation_words])

        # Positive/negative words
        self._add_counts("lexicon.pos_word", out, pos_words)
        self._add_counts("lexicon.neg_word", out, neg_words)

        # Negation scope words
        self._add_counts("negation.scope_word", out, scope_words)
        
        # Subjectivity marker words
        subj_marker_words = [w for w in tokens_no_stopwords if w in self.subjectivity_markers]
        self._add_counts("subjectivity.marker_word", out, subj_marker_words)

        return out

    # ==================== HELPER METHODS ====================

    def _add_counts(self, prefix: str, out: Dict[str, Any], words: list):
        """
        Add word counts to output dict with given prefix.
        
        Creates entries like "prefix=word: count" for DictVectorizer.
        """
        for w, c in Counter(words).items():
            out[f"{prefix}={w}"] = c
        return out

    def _negation_scope(self, tokens: List[str], neg_positions: List[int], window: int = 3) -> List[str]:
        """
        Extract words in the scope of negation (typically next N words).
        
        Args:
            tokens: Token list
            neg_positions: Indices where negation words occur
            window: Number of tokens after negation to consider
            
        Returns:
            List of tokens in negation scope (with duplicates)
        """
        negated_terms = []
        for pos in neg_positions:
            scope = tokens[pos + 1: pos + window + 1]
            negated_terms.extend(scope)
        return negated_terms

    def _count_sentiment_near_positions(self, tokens: List[str], positions: List[int], window: int = 3) -> int:
        """
        Count sentiment word occurrences near specified positions.
        
        Used to detect sentiment words near hedges (softened sentiment).
        """
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
        """
        Extract vocabulary words (with duplicates) near specified positions.
        
        Args:
            tokens: Token list
            positions: Target positions to search around
            window: Size of window (+/- tokens)
            vocab: Set of vocabulary words to match
            
        Returns:
            List of matching words with duplicates preserved
        """
        found = []
        for pos in positions:
            start = max(0, pos - window)
            end = min(len(tokens), pos + window + 1)
            for w in tokens[start:end]:
                if w in vocab:
                    found.append(w)
        return found

    def _compute_compositional_polarity(
        self,
        tokens: List[str],
        neg_positions: List[int],
        pos_words: List[str],
        neg_words: List[str]
    ) -> Dict[str, float]:
        """
        **NEW**: Compute compositionally adjusted sentiment polarity.
        
        This method captures how negation and intensifiers modify sentiment:
        
        1. **Negation flips**: Words following negation have reversed polarity
           - "not good" → counts as negative instead of positive
           - "not bad" → counts as positive instead of negative
        
        2. **Intensifier boost**: Sentiment words preceded by intensifiers
           get amplified weight (e.g., "very good" > "good")
        
        Args:
            tokens: Token list
            neg_positions: Indices of negation words
            pos_words: List of positive sentiment words found
            neg_words: List of negative sentiment words found
            
        Returns:
            Dict with adjusted counts and metrics:
            - adjusted_pos_count: positive count after negation/intensification
            - adjusted_neg_count: negative count after negation/intensification
            - adjusted_net_sentiment: net sentiment (positive - negative)
            - intensifier_boost: total boost from intensifiers
            - negation_flips: number of sentiment words flipped by negation
        """
        intensifiers = set(self.lexicon.intensifiers)
        pos_set = set(self.lexicon.positive_words)
        neg_set = set(self.lexicon.negative_words)
        
        # Track positions of sentiment words
        pos_word_positions = [i for i, w in enumerate(tokens) if w in pos_set]
        neg_word_positions = [i for i, w in enumerate(tokens) if w in neg_set]
        
        adjusted_pos = len(pos_words)
        adjusted_neg = len(neg_words)
        intensifier_boost = 0
        negation_flips = 0
        
        # Apply negation flips (within window of 3 tokens after negation)
        negation_window = 3
        for neg_pos in neg_positions:
            scope_range = range(neg_pos + 1, min(neg_pos + negation_window + 1, len(tokens)))
            
            for sent_pos in pos_word_positions:
                if sent_pos in scope_range:
                    adjusted_pos -= 1  # Remove from positive
                    adjusted_neg += 1  # Add to negative
                    negation_flips += 1
            
            for sent_pos in neg_word_positions:
                if sent_pos in scope_range:
                    adjusted_neg -= 1  # Remove from negative
                    adjusted_pos += 1  # Add to positive
                    negation_flips += 1
        
        # Apply intensifier boost (intensifier immediately before sentiment word)
        all_sent_positions = pos_word_positions + neg_word_positions
        for sent_pos in all_sent_positions:
            if sent_pos > 0 and tokens[sent_pos - 1] in intensifiers:
                intensifier_boost += 0.5  # Boost by 0.5 per intensifier
        
        # Adjust counts with boost
        adjusted_pos += intensifier_boost / 2  # Distribute boost
        adjusted_neg += intensifier_boost / 2
        
        return {
            "adjusted_pos_count": adjusted_pos,
            "adjusted_neg_count": adjusted_neg,
            "adjusted_net_sentiment": adjusted_pos - adjusted_neg,
            "intensifier_boost": intensifier_boost,
            "negation_flips": negation_flips
        }

    def _extract_subjectivity_markers(self, tokens: List[str], raw_text: str) -> Dict[str, float]:
        """
        **NEW**: Extract explicit subjectivity and opinion markers.
        
        Identifies phrases that signal subjective content:
        - First-person opinions: "I think", "I believe", "I feel", "in my opinion"
        - Epistemic markers: "probably", "possibly", "perhaps", "maybe"
        - Appearance markers: "it seems", "it appears", "apparently"
        
        These markers indicate opinion rather than fact, and suggest
        uncertainty or personal judgment.
        
        Args:
            tokens: Token list
            raw_text: Original text (for multi-word phrase matching)
            
        Returns:
            Dict with subjectivity marker counts and density
        """
        # Normalize text for pattern matching
        text_lower = raw_text.lower()
        
        # First-person opinion markers
        first_person_patterns = [
            'i think', 'i believe', 'i feel', 'in my opinion',
            'personally', 'i would say', 'i suppose'
        ]
        first_person_count = sum(text_lower.count(pattern) for pattern in first_person_patterns)
        
        # Epistemic (uncertainty) markers
        epistemic_markers = {
            'probably', 'possibly', 'perhaps', 'maybe', 'supposedly',
            'presumably', 'conceivably'
        }
        epistemic_count = sum(1 for t in tokens if t in epistemic_markers)
        
        # Appearance/seeming markers
        appearance_patterns = [
            'it seems', 'it appears', 'apparently', 'seemingly',
            'it looks like', 'seems like'
        ]
        appearance_count = sum(text_lower.count(pattern) for pattern in appearance_patterns)
        
        # Total unique markers found
        total_markers = first_person_count + epistemic_count + appearance_count
        marker_density = total_markers / max(len(tokens), 1)
        
        return {
            "marker_count": total_markers,
            "marker_density": marker_density,
            "first_person_opinion_count": first_person_count,
            "epistemic_count": epistemic_count,
            "appearance_count": appearance_count
        }

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
        """Extract syntactic complexity features."""
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
        """Extract sentence-level aggregated features."""
        num_sentences = len(sentences)
        sentence_sentiments = [self.sia.polarity_scores(s)["compound"] for s in sentences]
        avg_sentiment = np.mean(sentence_sentiments) if sentence_sentiments else 0
        variance = np.var(sentence_sentiments) if len(sentence_sentiments) > 1 else 0
        return {
            'num_sentences': num_sentences,
            'avg_sentiment': avg_sentiment,
            'sentiment_variance': variance,
        }