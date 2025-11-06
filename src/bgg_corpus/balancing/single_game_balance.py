import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from .helpers import create_augmented_review
from .augmentation import AugmentationManager
from ..resources import LOGGER
from ..models import CorpusDocument


def balance_single_game(
    documents: List[CorpusDocument],
    categories=('positive', 'neutral', 'negative'),
    min_samples_for_balance=30,
    balance_strategy='oversample',  # 'oversample', 'undersample', 'hybrid'
    target_ratio=None,
    augmentation_manager: Optional[AugmentationManager] = None,
    max_augmentations_per_review=2,
    verbose=False
) -> Tuple[List[CorpusDocument], Dict]:
    """
    Balance CorpusDocuments for a single game using text augmentation and/or subsampling.
    """

    # ----------------------------------------
    # 1. Group docs by category
    # ----------------------------------------
    cat_dict = defaultdict(list)
    for doc in documents:
        cat = getattr(doc.review, 'category', None)
        if cat in categories:
            cat_dict[cat].append(doc)

    counts_before = {cat: len(cat_dict[cat]) for cat in categories}
    max_count = max(counts_before.values()) if counts_before else 0
    min_count = min((c for c in counts_before.values() if c > 0), default=0)

    # ----------------------------------------
    # 2. Handle empty categories
    # ----------------------------------------
    if min_count == 0 or max_count == 0:
        msg = f"⚠️ Skipping balance: at least one category is empty. counts={counts_before}"
        if verbose:
            print(msg)
        LOGGER.info(msg)
        return documents, {
            'before': counts_before,
            'after': counts_before,
            'strategy': 'none',
            'balanced': False,
            'augmented_count': 0,
            'subsampled_count': 0,
        }

    # ----------------------------------------
    # 3. Determine target counts
    # ----------------------------------------
    if balance_strategy == 'undersample':
        base_target = min_count
    elif balance_strategy == 'oversample':
        base_target = max_count
    else:
        if target_ratio is None:
            ratio = max_count / max(min_count, 1)
            target_ratio = (
                0.5 if ratio > 10 else
                0.6 if ratio > 5 else
                0.75 if ratio > 2 else
                1.0
            )
        base_target = max(int(max_count * target_ratio), min_samples_for_balance)

    target_counts = {cat: base_target for cat in categories}

    # ----------------------------------------
    # 4. Perform balancing
    # ----------------------------------------
    balanced_docs = []
    counts_after = {}
    augmented_count = 0
    subsampled_count = 0
    augmentation_used = False

    for cat in categories:
        cat_docs = cat_dict[cat]
        current = len(cat_docs)
        target_count = target_counts[cat]

        if current == 0:
            counts_after[cat] = 0
            continue

        # --- A: Need to increase samples
        if current < target_count and augmentation_manager is not None:
            needed = target_count - current
            lang = getattr(cat_docs[0].review, 'language', 'en') or 'en'

            if verbose:
                print(f"  {cat}: augmenting (need {needed})")

            augmented_docs = []
            attempts = 0
            max_attempts = needed * 5

            while len(augmented_docs) < needed and attempts < max_attempts:
                attempts += 1
                base_doc = random.choice(cat_docs)
                base_review = base_doc.review

                try:
                    aug_texts = augmentation_manager.augment(
                        base_review.comment.strip(), lang=lang, num_augmentations=1
                    )
                except Exception as e:
                    LOGGER.debug(f"Augmentation error for review id={getattr(base_review, 'id', 'n/a')}: {e}")
                    continue

                if not aug_texts:
                    continue

                for aug_text in aug_texts if isinstance(aug_texts, list) else [aug_texts]:
                    if len(augmented_docs) >= needed:
                        break
                    aug_str = aug_text.strip()
                    if not aug_str or aug_str == base_review.comment.strip():
                        continue

                    aug_review = create_augmented_review(base_review, aug_str)
                    from ..models import CorpusDocument
                    aug_doc = CorpusDocument(
                        review=aug_review,
                        tokens=[],
                        lemmas=[],
                        language=aug_review.language or lang,
                        is_augmented=True
                    )
                    augmented_docs.append(aug_doc)
                    augmented_count += 1

            cat_docs = cat_docs + augmented_docs[:needed]
            counts_after[cat] = len(cat_docs)
            augmentation_used = True

        # --- B: Too many samples
        elif current > target_count and balance_strategy in ('undersample', 'hybrid'):
            removed = current - target_count
            cat_docs = random.sample(cat_docs, target_count)
            subsampled_count += removed
            if verbose:
                print(f"  {cat}: -{removed} subsampled")

        # --- Clean and add
        clean_docs = [d for d in cat_docs if getattr(d.review, 'comment', '').strip()]
        balanced_docs.extend(clean_docs)
        counts_after[cat] = len(clean_docs)

    random.shuffle(balanced_docs)

    stats = {
        'before': counts_before,
        'after': counts_after,
        'strategy': balance_strategy,
        'balanced': True,
        'augmented_count': augmented_count,
        'subsampled_count': subsampled_count,
        'augmentation_used': augmentation_used,
    }

    LOGGER.info(f"Balanced game: {counts_before} → {counts_after}")
    return balanced_docs, stats