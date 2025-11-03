[⬅ Back to bgg_corpus README](../README.md)

# Features Submodule

The `features` submodule provides tools for **feature extraction** from board game reviews. It supports **lexicon-based sentiment features, linguistic analysis, and vectorization** for machine learning.

```
features/
├── lexicons.py                # Sentiment and domain lexicons
├── linguistic_extractor.py    # Extract numeric and categorical linguistic features
├── vectorization.py           # TF-IDF and DictVectorizer for text and opinion features
└── __init__.py
```

## Lexicons

### `lexicons.py`

- Handles **loading and managing lexical resources** for board game reviews defined as `.txt` files in the [lexicons](../../../data/lexicons) directory.
- Includes:

  - **Sentiment words**: positive, negative
  - **Modifiers**: intensifiers, mitigators, negations
  - **Domain-specific terms**
  - **Hedges**: hedge words, propositional, relational, discourse markers

- Provides combined sets for fast access (e.g., `all_hedges`).
- These lexicons are used in downstream feature extraction to compute **counts, densities, and proximity-based metrics**.

## Linguistic Feature Extraction

### `linguistic_extractor.py`

The `LinguisticFeaturesExtractor` extracts a **rich set of linguistic and textual features** from processed reviews (`CorpusDocument`) and returns a **single dictionary** suitable for `DictVectorizer`. Features include numeric metrics, ratios, counts, sequence-of-strings features, and co-occurrences.

| Category                                 | Feature                                                                                             | Description                                        |
| ---------------------------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------------- |
| **Sentiment**                            | `sentiment.pos_count`                                                                               | Count of positive words                            |
|                                          | `sentiment.neg_count`                                                                               | Count of negative words                            |
|                                          | `sentiment.total`                                                                                   | Total positive + negative words                    |
|                                          | `sentiment.pos_ratio`                                                                               | Ratio of positive words to total sentiment words   |
|                                          | `vader.compound`                                                                                    | Vader compound score                               |
|                                          | `vader.pos` / `vader.neu` / `vader.neg`                                                             | Vader sentiment sub-scores                         |
|                                          | `sentiment.first_sentence` / `last_sentence`                                                        | Vader score of first/last sentence                 |
|                                          | `sentiment.max_sentence` / `min_sentence`                                                           | Max/min sentence-level sentiment                   |
| **Syntactic**                            | `syntactic.num_tokens_no_stop`                                                                      | Number of tokens after stopword removal            |
|                                          | `syntactic.avg_token_no_stop_length`                                                                | Average token length                               |
|                                          | `syntactic.avg_dep_depth`                                                                           | Average dependency tree depth                      |
|                                          | `syntactic.num_dependencies`                                                                        | Number of dependencies parsed                      |
| **Lexical Diversity**                    | `lexical.ttr`                                                                                       | Type-token ratio                                   |
|                                          | `lexical.hapax_ratio`                                                                               | Ratio of hapax legomena                            |
|                                          | `lexical.entropy`                                                                                   | Shannon entropy of token distribution              |
| **Sentence-Level**                       | `sentence.num_sentences`                                                                            | Number of sentences                                |
|                                          | `sentence.avg_sentiment`                                                                            | Average sentence sentiment                         |
|                                          | `sentence.sentiment_variance`                                                                       | Variance of sentence sentiment                     |
| **Readability / Complexity**             | `readability.fk_grade`                                                                              | Flesch-Kincaid grade level                         |
|                                          | `readability.ease`                                                                                  | Flesch reading ease                                |
|                                          | `readability.complex_word_ratio`                                                                    | Ratio of polysyllabic words                        |
| **Hedging**                              | `hedge.count`                                                                                       | Total number of hedge words                        |
|                                          | `hedge.count_type.hedge_words` / `propositional` / `relational` / `discourse_markers`               | Counts per hedge subtype                           |
|                                          | `hedge.density`                                                                                     | Hedges per token ratio                             |
|                                          | `hedge.nearby_sentiment_count`                                                                      | Number of sentiment words near any hedge           |
|                                          | `hedge.nearby_sentiment_ratio`                                                                      | Ratio of sentiment words near hedges               |
|                                          | `hedge.prop_propositional` / `prop_relational` / `prop_discourse`                                   | Proportions of each hedge type                     |
| **Negation**                             | `negation.count`                                                                                    | Number of negation words                           |
|                                          | `negation.sentiment_ratio`                                                                          | Ratio of negation words to total sentiment words   |
|                                          | `negation.scope_word`                                                                               | List of words within negation scope                |
| **Domain-Specific / Lexicons**           | `domain.<category>`                                                                                 | Words from domain-specific lexicons per category   |
|                                          | `lexicon.intensifier` / `mitigator` / `negation`                                                    | Counts of intensifiers, mitigators, negation words |
|                                          | `lexicon.pos_word` / `neg_word`                                                                     | Positive/negative words for DictVectorizer         |
| **Co-occurrences**                       | `sentiment_unigram_count`                                                                           | Count of sentiment words in unigrams               |
|                                          | `domain_sentiment_bigram_count`                                                                     | Count of domain + sentiment bigrams                |
|                                          | `domain_sentiment_trigram_count`                                                                    | Count of domain + sentiment trigrams               |
| **POS Ratios**                           | `pos.adj_ratio` / `adv_ratio` / `noun_ratio` / `verb_ratio`                                         | Ratio of adjectives, adverbs, nouns, verbs         |
| **Punctuation**                          | `punct.exclamation_count`                                                                           | Number of exclamation marks                        |
|                                          | `punct.repeated_punct_count`                                                                        | Count of repeated punctuation (`!!`, `??`, etc.)   |
| **Sequence-of-Strings (DictVectorizer)** | `hedge.words` / `hedge.propositional` / `hedge.relational` / `hedge.discourse_marker` / `hedge.all` | Lists of hedge words per type                      |
|                                          | `hedge.nearby_sentiment_words`                                                                      | Sentiment words appearing near hedges              |
|                                          | `lexicon.intensifier` / `mitigator` / `negation`                                                    | Lists of intensifiers, mitigators, negations       |
|                                          | `lexicon.pos_word` / `neg_word`                                                                     | Lists of positive/negative words                   |
|                                          | `negation.scope_word`                                                                               | Words within negation scopes                       |
|                                          | `domain.<category>`                                                                                 | Lists of domain-specific terms found in review     |

**Note:** All features are combined into a single dictionary so that **numeric features are stored as floats/ints** and **sequence-of-strings features are stored as lists**, ready for vectorization using `DictVectorizer` or integration into ML pipelines.

## Vectorization

### `vectorization.py`

The `ReviewVectorizer` combines **textual features (TF-IDF)** with **opinion/linguistic features (DictVectorizer)** into a single sparse matrix, ready for ML models. It is **multilingual-aware** by prefixing tokens with the document language.

| Feature / Method                                         | Description                                                                                                                                                                           |
| -------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Initialization**                                       | `ReviewVectorizer(max_features=5000, ngram_range=(1,2), stopwords=None)`<br>Creates TF-IDF and DictVectorizer instances.                                                              |
| `_prefix_tokens_with_language(tokens_per_doc, langs)`    | Adds a language prefix to each token to avoid conflicts across languages in TF-IDF.                                                                                                   |
| `fit_transform(tokens_per_doc, langs, opinion_features)` | Fits TF-IDF on token lists and DictVectorizer on opinion features. Returns a **combined sparse matrix** (TF-IDF + opinion features).                                                  |
| `transform(tokens_per_doc, langs, opinion_features)`     | Transforms new data using already-fitted TF-IDF and DictVectorizer. Returns combined sparse matrix.                                                                                   |
| **Textual Features (TF-IDF)**                            | - Tokens with language prefixes<br>- N-grams controlled by `ngram_range`<br>- Maximum features controlled by `max_features`<br>- Stopwords removed if provided                        |
| **Opinion / Linguistic Features (DictVectorizer)**       | - Numeric and sequence-of-strings features from `LinguisticFeaturesExtractor`<br>- Handles counts, ratios, and co-occurrences<br>- Preserves all keys as columns in the sparse matrix |
| **Output**                                               | Sparse matrix (`hstack`) combining TF-IDF and opinion features for downstream ML pipelines.                                                                                           |
| **Multilingual Awareness**                               | Prefixes tokens with language codes (`en_game`, `fr_game`) to avoid conflicts in multilingual corpora.                                                                                |

**Example Usage:**

```python
vectorizer = ReviewVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(tokens_per_doc, langs, opinion_features)
X_new = vectorizer.transform(new_tokens, new_langs, new_opinion_features)
```

## Summary

- `lexicons.py`: Load sentiment, hedges, and domain-specific lexicons from `BoardGeekGames-Corpus/data/lexicons`.
- `linguistic_extractor.py`: Compute **numeric, syntactic, lexical, and hedging features**.
- `vectorization.py`: Combine **text and opinion features** into a vector representation for ML pipelines.

This submodule enables **feature-rich, multilingual, and lexicon-aware representations** of board game reviews, ready for classification or regression tasks.
