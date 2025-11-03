[⬅ Back to bgg_corpus README](../README.md)

# Preprocessing Submodule

The `preprocessing` submodule provides tools to **clean, analyze, and prepare text reviews** for NLP tasks.  
It is organized into three conceptual layers, but in practice, all steps are **fused together in the `review_processor` pipeline**:

```
preprocessing/
├── cleaning.py            # Basic text cleaning utilities
├── review_processor.py    # Main high-level pipeline for processing reviews
├── spacy_analysis.py      # Advanced NLP analysis (POS, lemmas, entities, etc.)
├── language/              # Language detection and spaCy model management
└── tokenization/          # Sentence segmentation, word tokenization, stemming
```

> ⚠️ **Important:**  
> All the preprocessing is integrated into a single workflow via `process_review_item`.  
> The main function used to create **the whole corpus**, [`build_corpus(args)`](./utilities/corpus_builder.py), calls a **wrapper** of this function: [`process_single_review`](./utilities/processing_utils.py),  
> which **generates [`CorpusDocument`](./models/corpus_document.py) objects** ready for further processing, feature extraction, and storage.

## Structure Overview

| Layer / Module         | Role / Responsibilities                                                       |
| ---------------------- | ----------------------------------------------------------------------------- |
| **Core preprocessing** | `cleaning.py` – text normalization and basic cleaning                         |
| **Language handling**  | `language/` – language detection, spaCy model selection, stopwords mapping    |
| **Tokenization**       | `tokenization/` – sentence segmentation, word tokenization, stemming          |
| **NLP analysis**       | `spacy_analysis.py` – POS, lemmas, dependencies, named entities               |
| **Pipeline**           | `review_processor.py` – orchestrates all steps and generates `CorpusDocument` |

---

## Main Pipeline: `review_processor.py`

The high-level function `process_review_item` **wraps all steps** into a single processing call:

1. Clean raw text (`cleaning.py`)
2. Detect language (`language/detection.py`)
3. Segment sentences and tokenize (`tokenization/segmentation.py`)
4. Filter tokens (stopwords, non-alphanumeric)
5. Apply stemming/lemmatization (`tokenization/stemming.py`)
6. Extract linguistic features with the class defined in features module: [LinguisticFeaturesExtractor](./features/linguistic_extractor.py)
7. Aggregate all results into a structured dictionary for `CorpusDocument`

**Example usage:**

```python
from preprocessing.review_processor import process_review_item

item = {
    "username": "user123",
    "rating": 8,
    "raw_text": "I loved this game! The mechanics are fantastic.",
    "timestamp": "2023-10-01",
}

processed = process_review_item(item)
print(processed["tokens_no_stopwords"])
print(processed["linguistic_features"]["sentiment.pos_count"])
```

**Returned fields:**

| Key                   | Description                                             |
| --------------------- | ------------------------------------------------------- |
| `raw_text`            | Original review text                                    |
| `clean_text`          | Normalized text                                         |
| `language_detected`   | Detected ISO language code                              |
| `language`            | spaCy language code                                     |
| `sentences`           | Sentence segmentation                                   |
| `tokens`              | Tokenized text                                          |
| `tokens_no_stopwords` | Tokens filtered by stopwords                            |
| `stems`               | Stemmed tokens                                          |
| `lemmas`              | Lemmatized tokens                                       |
| `pos_tags`            | POS tags                                                |
| `dependencies`        | Dependency parses                                       |
| `entities`            | Named entities                                          |
| `linguistic_features` | Numeric and sequence features                           |
| `patterns`            | Extracted special patterns (emails, URLs, emojis, etc.) |

## Modules in Detail

### Core Preprocessing (`cleaning.py`)

- Text normalization: lowercasing, punctuation cleaning, removing unwanted characters.
- Forms the **first step** in the main pipeline.

### NLP Analysis (`spacy_analysis.py`)

- Extracts:

  - POS tags
  - Lemmas
  - Dependencies
  - Named entities

- Works per language via spaCy model selection in the language subpackage.

### Language Subpackage (`language/`)

- `detection.py`: Language detection using `langid` (ISO 639-1 codes)
- `spacy_utils.py`: Maps ISO codes to spaCy models and NLTK stopwords; lazy loads models with fallback.

### Tokenization Subpackage (`tokenization/`)

- `segmentation.py`: Sentence splitting (NLTK `punkt` + regex fallback)
- `stemming.py`: Stem tokens using Porter, Lancaster, Snowball, ISRI, RSLP
- `tokenizers_registry.py`: Central registry for sentence/word tokenizers and stemmers

## Summary

- All preprocessing modules are **fused in the main pipeline** `process_review_item`.
- **Multi-language support**, tokenization, stemming, and feature extraction are handled automatically.
- Generates structured outputs ready for **feature extraction**, **modeling**, or storage in `CorpusDocument`.
