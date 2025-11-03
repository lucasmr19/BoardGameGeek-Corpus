# 🧩 BoardGameGeek Corpus

**BoardGameGeek Corpus** is a Python project for building and analyzing an annotated textual corpus of **board game reviews**.  
It focuses on sentiment analysis, linguistic annotation, and lexicon-based modeling from user-generated content gathered from [BoardGameGeek](https://boardgamegeek.com).

## 🚀 Overview

This project automates the **collection, processing, and annotation** of board game reviews to create a reusable **linguistic corpus** for NLP and sentiment classification tasks.

- **Corpus construction** from multiple sources (crawler/API).
- **Text preprocessing**: cleaning, normalization, tokenization, lemmatization, POS tagging.
- **Linguistic annotation**: sentiment, negations, intensifiers, domain terms, hedges.
- **Balanced datasets** for supervised sentiment classification.
- **Vectorization and modeling**: TF-IDF, opinion features, and classifiers.

For detailed descriptions of modules, see the respective [`README.md`](./src/bgg_corpus/README.md) files.

## 📁 Project Structure

```
BoardGameGeek-Corpus/
├── README.md
├── requirements.txt
├── data/
│   ├── api/                  # API metadata JSONs
│   ├── crawler/              # Crawler reviews JSONs and stats
│   ├── lexicons/             # Sentiment, hedge, domain lexicons
│   ├── processed/            # Balanced corpora, datasets, vectors, models
│   │   ├── balance_reports/
│   │   ├── corpora/
│   │   │   ├── bgg_corpus.json
│   │   │   └── statistics/    # Corpus statistics and figures
│   │   │       ├── corpus_statistics_report.txt
│   │   │       └── figures/
│   │   │           ├── lexical_dispersion.png
│   │   │           ├── word_frequency_distribution.png
│   │   │           └── word_length_distribution.png
│   │   ├── datasets/         # Train/val/test splits
│   │   ├── models/           # Trained models & summaries
│   │   └── vectors/          # TF-IDF and opinion feature matrices
│   └── raw/                  # Original dump CSV from BGG page: https://boardgamegeek.com/data_dumps/bg_ranks
├── docs/                     # Diagrams and figures
├── notebooks/
├── src/
│   └── bgg_corpus/           # Core Python package
│       ├── README.md
│       ├── preprocessing/    # Cleaning, tokenization, spaCy analysis
│       ├── features/         # Lexicons, vectorization
│       ├── models/           # Corpus and document classes
│       ├── utilities/        # Helpers, corpus builder
│       ├── balancing/        # Oversampling/undersampling/augmentation
│       ├── storage/          # MongoDB storage
│       └── downloaders/      # Crawler/API downloaders
│   └── scripts/              # Executable scripts post-corpus creation
└── tests/
```

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/lucasmr19/BoardGameGeek-Corpus.git
cd BoardGameGeek-Corpus
pip install -r requirements.txt
```

## 🛠 Scripts Overview

All executable scripts are supposed to run post-corpus creation, see here [scripts README](./src/scripts/README.md)

| Script                 | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `pln_p2_7462_02_e1.py` | Extract linguistic features (sentiment, negations, intensifiers, hedges, domain terms) |
| `pln_p2_7462_02_e2.py` | Vectorization (TF-IDF n-grams + opinion features)                                      |
| `pln_p2_7462_02_e3.py` | Dataset creation (train/val/test splits)                                               |
| `pln_p2_7462_02_e4.py` | Classification model training (NB, SVM, RF, XGBoost)                                   |
| `pln_p2_7462_02_e5.py` | Model evaluation and technical report generation                                       |

## 🧠 Key Components

- **Corpus Construction:** [`corpus_builder.py`](./src/bgg_corpus/utilities/corpus_builder.py) – handles aggregation of raw data into structured corpus objects.
- **Preprocessing:** [`processing_utils.py`](./src/bgg_corpus/utilities/processing_utils.py) – text cleaning, normalization, tokenization, and lemmatization.
- **Corpus Objects:** [`CorpusDocument`](./src/bgg_corpus/models/corpus_document.py) – core object representing a single review with annotations.
- **Feature Extraction:** [`linguistic_extractor.py`](./src/bgg_corpus/features/linguistic_extractor.py) – extracts sentiment, lexical, and syntactic features.
- **Storage:** [`mongodb_storage.py`](./src/bgg_corpus/storage/mongodb_storage.py) – optional persistence layer for storing/retrieving corpora in MongoDB.
- **Scripts:** Executables for feature extraction, vectorization, dataset creation, modeling, and evaluation.

> Each subpackage contains a `README.md` explaining its purpose, usage, and examples.

## 📌 Project Goals

- Build a **domain-specific sentiment corpus** from BoardGameGeek reviews.
- Extract and annotate **linguistic and lexical features** for NLP tasks.
- Provide **structured datasets** for supervised sentiment classification.
- Enable **scalable and extensible analysis** for research or downstream applications.

## ⚡ Usage Notes

1. **Corpus creation** must be completed before running any script in `src/scripts/`.
2. **Scripts are independent** but rely on the preprocessed corpus JSON in `data/processed/corpora/bgg_corpus.json`.
3. Each script can accept optional parameters (paths, feature selection, splits, etc.). See individual examples in the [scripts README](./src/scripts/README.md).
4. Generated outputs (features, vectors, models, evaluation reports) are stored in the corresponding `data/processed/` subdirectories.

## License

This project is licensed under the [MIT License](LICENSE).
