[⬅ Back to bgg_corpus README](../README.md)

# Utilities Submodule

The `utilities` submodule provides helper functions to **load, process, merge, and build structured corpora** of board game reviews.

> ⚠️ **Main Pipeline:** The central entry point is the [`build_corpus`](./utilities/corpus_builder.py) function, which handles review collection, balancing, and preprocessing via the **preprocessing pipeline**.  
> This pipeline internally calls the **wrapper** [`process_single_review`](./utilities/processing_utils.py), which converts raw reviews into fully processed **[`CorpusDocument`](./models/corpus_document.py) objects**, ready for feature extraction, modeling, or storage.

```

utilities/
├── corpus_builder.py      # ensure_review_obj, build_corpus (main pipeline)
├── io_utils.py            # load_json, load_csv
├── metadata_utils.py      # build_metadata, load_metadata_from_api
├── processing_utils.py    # process_single_review (preprocessing wrapper)
├── review_utils.py        # merge_reviews
├── **init**.py

```

---

## 1. Corpus Builder (`corpus_builder.py`) – Main Pipeline

- **Purpose:** Build a complete, structured, and optionally balanced corpus from raw reviews.
- **Key Functions:**

  - `ensure_review_obj(r, gid)` → Standardizes review format.
  - `build_corpus(game_ids, **kwargs)` → Main pipeline.

- **Workflow:**

  1. Collect and optionally balance reviews across multiple games.
  2. Preprocess each review via `process_single_review` (wrapper of `review_processor.py`).
  3. Assemble the corpus hierarchy: `Corpus` → `GameCorpus` → `CorpusDocument`.

- **Parameters:** `source`, `balance_strategy`, `use_augmentation`, `parallel`, `preprocess_kwargs`, etc.

- **Example:**

```python
from utilities.corpus_builder import build_corpus

corpus, stats = build_corpus(
    game_ids=[50, 51, 52],
    source="combined",
    balance_strategy="hybrid",
    use_augmentation=True,
    parallel=True,
    max_workers=4
)
```

---

## 2. Processing Utilities (`processing_utils.py`)

- **Purpose:** Wraps the **preprocessing pipeline** to convert raw reviews into `CorpusDocument` objects.

- **Main Function:** `process_single_review(review_item, **preprocess_kwargs)`

- **Example:**

```python
from utilities.processing_utils import process_single_review

doc = process_single_review(review_dict, lowercase=True, remove_punctuation=True)
```

---

## 3. Review Merging Utilities (`review_utils.py`)

- **Purpose:** Merge reviews from API and crawler sources into a consistent dataset.
- **Main Function:** `merge_reviews(game_id, source="combined")`
- **Merging Strategy:** Deduplicate by `(username + normalized comment)`; prioritize crawler timestamps.

---

## 4. Metadata Utilities (`metadata_utils.py`)

- **Purpose:** Aggregate and structure metadata for each game.
- **Main Functions:**

  - `load_metadata_from_api(game_id)`
  - `build_metadata(game_id)` → Returns `game_info`, `stats`, `rankings`, `classifications`, `polls`.

---

## 5. IO Utilities (`io_utils.py`)

- **Purpose:** Load external data files safely.
- **Functions:**

  - `load_json(path)` → Returns dict or `None`.
  - `load_csv(path)` → Returns `pandas.DataFrame` or empty DataFrame.

---

## Summary

| Module               | Role                                                                |
| -------------------- | ------------------------------------------------------------------- |
| **Corpus Builder**   | Main pipeline to collect, balance, preprocess, and assemble corpus. |
| **Processing Utils** | Wraps `review_processor.py` for single review preprocessing.        |
| **Review Utils**     | Merge API and crawler reviews.                                      |
| **Metadata Utils**   | Aggregate structured game metadata.                                 |
| **IO Utils**         | Safe loading of JSON/CSV files.                                     |

> ⚠️ **Important:** All text preprocessing and NLP transformations rely on the **preprocessing pipeline (`review_processor.py`)**.
> The `build_corpus` pipeline is the backbone for creating structured, balanced, and analysis-ready corpora.
