[⬅ Back to bgg_corpus README](../README.md)

# Balancing Submodule

The `balancing` submodule provides **corpus balancing and data augmentation tools** for board game reviews. It integrates **synthetic text generation** using NLP augmentation techniques to address class imbalance.

```
balancing/
├── augmentation.py          # NLP-based augmentation manager
├── helpers.py               # Helper functions for review augmentation
├── single_game_balance.py   # Balance reviews for a single game
├── multi_game_balance.py    # Apply balancing across multiple games
├── save_balance.py          # Save balancing reports
└── __init__.py
```

---

## NLP Augmentation

### `augmentation.py`

- Implements **`AugmentationManager`**, which provides text augmentation using **`nlpaug`**.
- Supported augmentation strategies:

  - **English (`en`)**: `SynonymAug` using WordNet.
  - **Other languages (`es, fr, de, it, pt, nl`)**: `BackTranslationAug` via Helsinki-NLP translation models.

- API example:

```python
augmenter = AugmentationManager()
augmented_texts = augmenter.augment("The game was exciting!", lang='en', num_augmentations=3)
```

- Returns **only augmented variants**, without including the original text.
- Automatically deduplicates and filters empty/identical outputs.

---

### `helpers.py`

- Provides utilities for **creating augmented review objects**:

  - Copies metadata from original reviews.
  - Marks augmented reviews with `is_augmented=True` and links to the original via `augmented_from`.

- Example:

```python
aug_review = create_augmented_review(original_review, "The game was thrilling!")
```

---

## Single-Game Balancing

### `single_game_balance.py`

- Balances sentiment categories (`positive`, `neutral`, `negative`) for **one game**.
- Supports multiple strategies:

  - **Oversample**: Increase minority classes.
  - **Undersample**: Reduce majority classes.
  - **Hybrid**: Combination using a target ratio.

- Handles **neutral reviews with double weight** for improved category balance.
- Can integrate `AugmentationManager` to create **synthetic reviews** for underrepresented classes.
- Returns:

  - **Balanced reviews** list.
  - **Statistics** about augmentation, subsampling, and category distribution.

---

## Multi-Game Balancing

### `multi_game_balance.py`

- Applies balancing across **multiple games**.
- Integrates:

  - Single-game balancing (`balance_single_game`)
  - Optional augmentation per game
  - Collection of **global statistics**:

    - Total reviews before/after
    - Counts per category
    - Augmented vs subsampled counts
    - Final max/min category ratio

- Supports verbose logging and detailed progress reporting.

---

## Reporting

### `save_balance.py`

- Saves **balance reports** as JSON files in a designated reports directory.
- Tracks:

  - Counts before and after balancing
  - Strategy used
  - Augmented/subsampled numbers per game

- Timestamped filenames for versioning.

---

## Summary

- `augmentation.py`: Synthetic review generation with **BackTranslation** or **Synonym replacement**.
- `helpers.py`: Utilities for creating augmented Review objects.
- `single_game_balance.py`: Balance one game’s reviews via oversampling, undersampling, or hybrid.
- `multi_game_balance.py`: Apply balancing across multiple games and collect **global stats**.
- `save_balance.py`: Store JSON reports with balancing statistics.

This submodule enables **robust class balancing for sentiment-labeled reviews**, supporting multilingual augmentation and integration into ML pipelines.
