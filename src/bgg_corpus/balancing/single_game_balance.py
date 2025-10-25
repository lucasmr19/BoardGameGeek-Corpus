import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from .helpers import create_augmented_review
from .augmentation import AugmentationManager
from ..resources import LOGGER


def balance_single_game(
    reviews,
    categories=('positive', 'neutral', 'negative'),
    min_samples_for_balance=30,
    balance_strategy='oversample',  # 'oversample', 'undersample', 'hybrid'
    target_ratio=None,
    augmentation_manager: Optional[AugmentationManager] = None,
    max_augmentations_per_review=2,
    verbose=False
) -> Tuple[List, Dict]:
    """
    Balance reviews for a single game using text augmentation and/or subsampling.

    The function ensures that:
        - Sentiment categories (positive, neutral, negative) are balanced according to the chosen strategy.
        - Neutral ratings (5–6) are given *double weight*, meaning that their target sample size
          will be twice that of other categories, to improve intra/inter-category balance.

    Balancing strategies:
        - "oversample": Increase minority classes (and neutral category more strongly).
        - "undersample": Reduce majority classes.
        - "hybrid": Combine both approaches using `target_ratio`.

    If an `AugmentationManager` is provided, it is used to create synthetic reviews for
    underrepresented categories through text augmentation.

    Args:
        reviews (list): List of Review objects to balance.
        categories (tuple): Sentiment categories to consider.
        min_samples_for_balance (int): Minimum number of samples required for balancing.
        balance_strategy (str): One of {"oversample", "undersample", "hybrid"}.
        target_ratio (float, optional): Ratio (minority/majority) for hybrid balancing.
        augmentation_manager (AugmentationManager, optional): Augmentation manager.
        max_augmentations_per_review (int): Max augmentations per base review.
        verbose (bool): If True, print detailed logs.

    Returns:
        Tuple[List[Review], Dict]:
            - balanced_reviews: The balanced list of Review objects.
            - stats: Dictionary with detailed balancing statistics.
    """

    # ----------------------------------------
    # 1. Group reviews by category
    # ----------------------------------------
    cat_dict = defaultdict(list)
    for r in reviews:
        cat = getattr(r, 'category', None)
        if cat in categories:
            cat_dict[cat].append(r)

    counts_before = {cat: len(cat_dict[cat]) for cat in categories}
    max_count = max(counts_before.values()) if counts_before else 0
    min_count = min((c for c in counts_before.values() if c > 0), default=0)

    # ----------------------------------------
    # 2. Handle empty categories
    # ----------------------------------------
    if min_count == 0 or max_count == 0:
        if verbose:
            print("⚠️ Skipping balance: at least one category is empty.")
        LOGGER.info("Skipping balance: at least one category is empty.")
        return reviews, {
            'before': counts_before,
            'after': counts_before,
            'strategy': 'none',
            'balanced': False,
            'augmented_count': 0,
            'subsampled_count': 0
        }

    # ----------------------------------------
    # 3. Determine base target count per strategy
    # ----------------------------------------
    if balance_strategy == 'undersample':
        base_target = min_count
    elif balance_strategy == 'oversample':
        base_target = max_count
    else:  # hybrid
        if target_ratio is None:
            ratio = max_count / max(min_count, 1)
            if ratio > 10:
                target_ratio = 0.5
            elif ratio > 5:
                target_ratio = 0.6
            elif ratio > 2:
                target_ratio = 0.75
            else:
                target_ratio = 1.0
        base_target = max(int(max_count * target_ratio), min_samples_for_balance)

    # ----------------------------------------
    # 4. Adjust target for neutral category (double size)
    # ----------------------------------------
    target_counts = {}
    for cat in categories:
        target_counts[cat] = base_target

    # ----------------------------------------
    # 5. Perform balancing
    # ----------------------------------------
    balanced_reviews = []
    counts_after = {}
    augmented_count = 0
    subsampled_count = 0
    augmentation_used = False

    for cat in categories:
        cat_reviews = cat_dict[cat]
        current = len(cat_reviews)
        target_count = target_counts[cat]

        if current == 0:
            counts_after[cat] = 0
            continue

        # --- Case A: Need to increase samples ---
        if current < target_count:
            needed = target_count - current
            if augmentation_manager is not None:
                lang = getattr(cat_reviews[0], 'language', 'en') or 'en'
                LOGGER.info(f"Augmenting category='{cat}' (current={current}, needed={needed}, lang={lang})")
                if verbose:
                    print(f"  {cat}: augmenting (need {needed})")

                augmented_reviews = []
                attempts = 0
                max_attempts = needed * 5  # avoid infinite loops

                while len(augmented_reviews) < needed and attempts < max_attempts:
                    attempts += 1
                    base_review = random.choice(cat_reviews)
                    per_review_try = min(max_augmentations_per_review, 1 + needed // max(1, current))

                    try:
                        aug_texts = augmentation_manager.augment(
                            base_review.comment, lang=lang, num_augmentations=per_review_try
                        )
                    except Exception as e:
                        LOGGER.debug(f"Augmentation error for review id={getattr(base_review, 'id', 'n/a')}: {e}")
                        continue

                    if not aug_texts:
                        continue

                    for aug_text in aug_texts if isinstance(aug_texts, list) else [aug_texts]:
                        if len(augmented_reviews) >= needed:
                            break
                        if not isinstance(aug_text, str):
                            continue
                        aug_str = aug_text.strip()
                        if not aug_str or aug_str == base_review.comment.strip():
                            continue

                        aug_review = create_augmented_review(base_review, aug_str)
                        augmented_reviews.append(aug_review)
                        augmented_count += 1

                        if verbose:
                            print(f"    + augmented from user={getattr(base_review, 'username', 'n/a')} "
                                  f"({len(augmented_reviews)}/{needed})")

                if len(augmented_reviews) < needed:
                    LOGGER.warning(f"⚠️ Could not fully augment category '{cat}' "
                                   f"({len(augmented_reviews)}/{needed})")

                cat_reviews = cat_reviews + augmented_reviews[:needed]
                counts_after[cat] = len(cat_reviews)
                augmentation_used = True

                LOGGER.info(f"Category '{cat}' augmented +{len(augmented_reviews[:needed])} "
                            f"(needed {needed}, attempts {attempts}).")

            else:
                LOGGER.warning(f"No augmentation manager available for '{cat}', skipping augmentation.")
                counts_after[cat] = current

        # --- Case B: Too many samples (undersampling/hybrid) ---
        elif current > target_count and balance_strategy in ('undersample', 'hybrid'):
            removed = current - target_count
            cat_reviews = random.sample(cat_reviews, target_count)
            subsampled_count += removed
            if verbose:
                print(f"  {cat}: -{removed} subsampled")
            LOGGER.info(f"Category '{cat}' subsampled -{removed}.")

        balanced_reviews.extend(cat_reviews)
        counts_after[cat] = len(cat_reviews)

    # ----------------------------------------
    # 6. Shuffle and return stats
    # ----------------------------------------
    random.shuffle(balanced_reviews)

    stats = {
        'before': counts_before,
        'after': counts_after,
        'strategy': balance_strategy,
        'balanced': True,
        'augmented_count': augmented_count,
        'subsampled_count': subsampled_count,
        'augmentation_used': augmentation_used
    }

    if verbose:
        LOGGER.info(
            f"Finished balancing: before={counts_before} after={counts_after} "
            f"augmented={augmented_count} subsampled={subsampled_count}"
        )

    return balanced_reviews, stats