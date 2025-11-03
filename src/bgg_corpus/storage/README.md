[⬅ Back to bgg_corpus README](../README.md)

# Storage Submodule

This submodule provides functionality for persisting and retrieving `Corpus` data in a database.

## Overview

Currently, the only implemented storage backend is:

- **MongoCorpusStorage**:
  Uses MongoDB to save and load `Corpus` objects. It is fully compatible with the `Corpus` class in `models/Corpus.py`.

### Features

- Save and load complete corpus data.
- Supports indexing and sorting via MongoDB.
- Simple to extend for other storage backends (e.g., SQL, flat files).

### Usage Example

```python
from src.bgg_corpus.storage.mongodb_storage import MongoCorpusStorage
from src.bgg_corpus.models.corpus import Corpus

# Initialize storage
storage = MongoCorpusStorage(db_name="bgg_corpus")

# Save corpus
corpus = Corpus.load_from_file("data/processed/corpora/bgg_corpus.json")
storage.save_corpus(corpus)

# Load corpus
loaded_corpus = storage.load_corpus()
```

### Initialization Parameters

| Parameter  | Type          | Default       | Description           |
| ---------- | ------------- | ------------- | --------------------- |
| `db_name`  | str           | `CORPUS_NAME` | MongoDB database name |
| `host`     | str           | `"localhost"` | MongoDB host          |
| `port`     | int           | `27017`       | MongoDB port          |
| `username` | Optional[str] | `None`        | MongoDB username      |
| `password` | Optional[str] | `None`        | MongoDB password      |

### Extensibility

New storage backends can be implemented by following the interface of `MongoCorpusStorage`. Any custom class should implement:

- `save_corpus(corpus: Corpus) -> None`
- `load_corpus() -> Corpus`

This ensures compatibility with the `Corpus` class and the rest of the project pipeline.
