[⬅ Back to bgg_corpus README](../README.md)

# Resources Submodule

This submodule provides **global runtime resources** used across the project, including language mappings, stopwords, and logging configuration.

## Overview

The `resources` submodule centralizes all project-wide constants and utilities, so other modules don’t need to redefine them.

### Features

- **Language mappings** for NLTK and spaCy.
  Helps select the correct language models and stopwords depending on the text language.

- **Stopwords cache**
  Automatically loads stopwords for all supported languages and caches them for quick access.

- **Logger**
  Standard logging configuration used across all modules.

## Language Mappings

| Mapping          | Description                                                                   |
| ---------------- | ----------------------------------------------------------------------------- |
| `NLTK_LANG_MAP`  | Maps ISO language codes (`en`, `es`, etc.) to NLTK language names.            |
| `SPACY_LANG_MAP` | Maps ISO language codes to spaCy model names (multiple options per language). |
| `SPACY_MODELS`   | Lazy-loaded cache for spaCy models.                                           |

## Stopwords

Stopwords are loaded from NLTK and cached in `STOPWORDS_CACHE`:

```python
from src.bgg_corpus.resources import STOPWORDS_CACHE

english_stopwords = STOPWORDS_CACHE["english"]
spanish_stopwords = STOPWORDS_CACHE["spanish"]
```

## Logger

A global logger is preconfigured:

```python
from src.bgg_corpus.resources import LOGGER

LOGGER.info("This is an info message")
LOGGER.error("This is an error message")
```

It uses the format:

```
YYYY-MM-DD HH:MM:SS [LEVEL] message
```

## Usage Example

```python
from src.bgg_corpus.resources import NLTK_LANG_MAP, STOPWORDS_CACHE, LOGGER

lang = "en"
stopwords_set = STOPWORDS_CACHE[NLTK_LANG_MAP[lang]]
LOGGER.info(f"Loaded {len(stopwords_set)} stopwords for {lang}")
```
