[⬅ Back to bgg_corpus README](../README.md)

# Models Submodule

The `models` submodule defines the **core POO hierarchy** for representing board game reviews and corpora in the project. These classes serve as the backbone for preprocessing, analysis, and storage.

## Class Hierarchy

```
Corpus
└── GameCorpus
    └── CorpusDocument
        └── Review
```

Each level wraps the one below, adding additional structure, metadata, and processing capabilities.

## Review

`Review` represents a **single review** from a user on BoardGameGeek.

**Attributes:**

- `username`: User who posted the review.
- `rating`: Numeric rating of the game.
- `comment`: Raw textual review.
- `timestamp`: Review timestamp.
- `game_id`: ID of the reviewed game.
- `label` / `category`: Categorical sentiment derived from `rating`.

**Purpose:**
Encapsulates the raw review and provides basic labeling. Includes serialization/deserialization utilities.

## CorpusDocument

`CorpusDocument` represents a **processed review**.

**Attributes:**

- `review`: The underlying `Review` object.
- `raw_text` / `clean_text`: Raw and cleaned versions of the text.
- `language`: Detected language of the review.
- `processed`: Dictionary containing NLP-extracted features:

  - `tokens`, `tokens_no_stopwords`, `stems`, `lemmas`
  - `pos_tags`, `dependencies`, `entities`, `linguistic_features`

- `patterns`: Detected patterns like emails, dates, hashtags, URLs, emojis, etc.
- `game_id`, `category`: Inherited from `Review`.

**Purpose:**
Adds **linguistic features** and preprocessing metadata to a review, bridging raw text and structured analysis.

## GameCorpus

`GameCorpus` represents a **collection of `CorpusDocument` objects** for a specific game.

**Attributes:**

- `game_id`: BoardGameGeek game ID.
- `metadata`: Optional game metadata (e.g., name, genre).
- `documents`: List of `CorpusDocument` objects.

**Methods:**

- `add_document(doc)`: Add a processed document to the corpus.
- `count_by_category()`: Count reviews per sentiment category.
- `to_dict() / from_dict()`: Serialize/deserialize the corpus.

**Purpose:**
Groups processed reviews by game and allows aggregation, counting, and structured access.

## Corpus

`Corpus` represents the **entire dataset** as a collection of `GameCorpus` objects. The methods are mostly inspired
by NLTK’s corpus interface.

**Attributes:**

- `games`: List of `GameCorpus` objects.
- `documents`: Flat property returning all `CorpusDocument` objects across games.

## Corpus Methods Overview

| Category                  | Method / Property                                      | Description                                                     |
| ------------------------- | ------------------------------------------------------ | --------------------------------------------------------------- |
| **I/O**                   | `from_json(path)`                                      | Load corpus from JSON                                           |
|                           | `to_json(path, list_format=True)`                      | Save corpus to JSON                                             |
|                           | `to_list_of_games()`                                   | Return corpus as a list of games with reviews                   |
|                           | `save_to_mongo(storage)`                               | Store corpus in MongoDB                                         |
|                           | `load_from_mongo(storage, ...)`                        | Load corpus from MongoDB                                        |
| **Access**                | `documents`                                            | Flat list of all `CorpusDocument`s                              |
|                           | `_select(game_ids=None, categories=None, labels=None)` | Filter documents                                                |
|                           | `metadata_for(game_id)`                                | Get game metadata                                               |
|                           | `all_classifications(field)`                           | Flatten classifications ('mechanics', 'categories', 'families') |
|                           | `classifications_by_game(game_id)`                     | Classifications for one game                                    |
| **Filtering**             | `filter_by_language(lang)`                             | Filter docs by language                                         |
|                           | `filter_by_game(game_ids)`                             | Filter docs by game ID                                          |
|                           | `filter_by_label(labels)`                              | Filter docs by label/category                                   |
| **Text / NLP**            | `raw(i=None)`                                          | Get raw text                                                    |
|                           | `raw_join(game_ids=None, categories=None)`             | Join raw texts                                                  |
|                           | `words(game_ids=None, categories=None)`                | Return token list                                               |
|                           | `sents(i=None)`                                        | Return sentences                                                |
|                           | `contexts(word, window=5)`                             | Left/right contexts of a word                                   |
|                           | `common_contexts(words, window=1)`                     | Most common contexts                                            |
|                           | `ngrams(n=2)` / `bigrams()` / `trigrams()`             | n-grams                                                         |
|                           | `lexical_diversity(tokens)`                            | Type-token ratio                                                |
| **Statistics**            | `ratings()`                                            | List of ratings                                                 |
|                           | `review_counts_by_category()`                          | Count by sentiment category                                     |
|                           | `rating_distribution(game_id=None)`                    | Ratings histogram                                               |
|                           | `avg_rating(game_id=None)`                             | Average rating                                                  |
|                           | `bayes_average(game_id=None)`                          | Bayesian average rating                                         |
|                           | `num_reviews(game_id=None)`                            | Total reviews                                                   |
|                           | `num_unique_users()`                                   | Unique user count                                               |
| **Visualization / Plots** | `lexical_dispersion_plot(words)`                       | Lexical dispersion                                              |
|                           | `plot_word_length_distribution()`                      | Word length distribution                                        |
|                           | `plot_frequency_distribution(n=30)`                    | Most frequent words                                             |
| **Aggregates / Metadata** | `overall_rank(game_id)` / `strategy_rank(game_id)`     | Game rankings                                                   |
|                           | `complexity_poll_avg(game_id)`                         | Complexity poll average                                         |

**Purpose:**
Provides a **global view** of all games and reviews, supports convenient access, filtering, and JSON I/O.

## Summary

- **`Review`** → Basic review object.
- **`CorpusDocument`** → Processed review with NLP features.
- **`GameCorpus`** → Collection of processed reviews for a single game.
- **`Corpus`** → Entire dataset, providing flat access and filtering utilities.

This hierarchy allows the project to handle reviews in a **modular, extensible, and structured way**, ready for feature extraction, modeling, or storage in MongoDB.
