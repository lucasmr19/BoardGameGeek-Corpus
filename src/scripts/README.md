[⬅ Back to BoardGameGeek README](../../README.md)

# BoardGameGeek Scripts Directory

This directory contains the main executable scripts for processing and analyzing
BoardGameGeek reviews. The scripts implement the workflow for feature extraction,
vectorization, dataset creation, model training, and evaluation.

## 1. `extract_linguistic_features.py`

**Purpose:**  
Extracts linguistic features such as sentiment, negation, intensifiers, domain terms,
and hedges from preprocessed BoardGameGeek reviews.

**Notes:**

- Relies on lexicons and utilities in `src/bgg_corpus/features/SentimentLexicon.py`.
- Redundant with the main preprocessing pipeline (`review_processor.py`), but provided for standalone execution.

**Usage Example:**

```bash
python extract_linguistic_features.py \
    --corpus data/processed/corpora/bgg_corpus.json \
    --output_dir data/processed/corpora/
```

**Output:**

- A corpus JSON file with linguistic features added (`<CORPUS_NAME>_features.json`).

## 2. `pln_p2_7462_02_e2.py`

**Purpose:**
Generates vector representations of BoardGameGeek reviews using:

- TF-IDF n-grams (unigrams + bigrams)
- Opinion / sentiment features
- Combined sparse representations

**Usage Example:**

```bash
python pln_p2_7462_02_e2.py \
    --corpus path/to/bgg_corpus.json \
    --output_dir path/to/save/vectors \
    --max_features 8000 \
    --ngram_range 1 2
```

**Output:**

- TF-IDF vector files, opinion/sentiment feature vectors, and combined vectors.

## 3. `pln_p2_7462_02_e3.py`

**Purpose:**
Creates train, validation, and test datasets for sentiment classification
from a preprocessed and labeled corpus.

**Usage Example:**

```bash
python pln_p2_7462_02_e3.py \
    --corpus_path data/bgg_corpus.json \
    --output_dir data/processed/datasets \
    --train_ratio 0.7 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --seed 42 \
    --format json \
    --verbose
```

**Output:**

- `train.json`, `val.json`, and `test.json` datasets for modeling.

## 4. `pln_p2_7462_02_e4.py`

**Purpose:**
Trains supervised classification models to predict review polarity using vector representations.

**Models Implemented:**

- Multinomial Naive Bayes
- Support Vector Machines (LinearSVC)
- Random Forest
- XGBoost (if available)

**Feature Subsets:**

- N-grams only (TF-IDF)
- Linguistic/sentiment features only
- Combined features (both)

**Usage Example:**

```bash
python pln_p2_7462_02_e4.py \
    --vector_dir path/to/vectors \
    --output_dir path/to/models
```

**Output:**

- Trained model files for each algorithm and feature subset.

## 5. `pln_p2_7462_02_e5.py`

**Purpose:**
Evaluates trained classification models and generates a technical report.

**Evaluation Includes:**

- Testing on train/test/validation splits
- Hyperparameter tuning (GridSearchCV or RandomizedSearchCV)
- Classification metrics: accuracy, precision, recall, F1-score
- Confusion matrices for best models
- Comprehensive technical report generation

**Usage Example:**

```bash
python pln_p2_7462_02_e5.py \
    --vector_dir path/to/vectors \
    --dataset_dir path/to/datasets \
    --models_dir path/to/models \
    --output_dir path/to/results
```

**Output:**

- Evaluation results, confusion matrices, and a technical report.

### Notes

- All scripts are Python 3 executables and require the project dependencies listed in `requirements.txt`.
- The scripts assume the project directory structure is preserved as in the `BoardGameGeek-Corpus` repository.
- Recommended to execute scripts in the order: **E1 → E2 → E3 → E4 → E5**.
