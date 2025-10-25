from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple

from .augmentation import AugmentationManager
from .single_game_balance import balance_single_game

def collect_balanced_reviews_multi_game(
    all_game_ids,
    merge_reviews_func,
    categories=('positive', 'neutral', 'negative'),
    min_samples_for_balance=30,
    balance_strategy='hybrid',  # 'oversample', 'undersample', 'hybrid'
    target_ratio=0.6,
    use_augmentation=True,
    max_augmentations_per_review=2,
    source="combined",
    verbose=True
) -> Tuple[List, Dict]:
    """
    Apply multi-game balancing with augmentation and/or subsampling.
    """
    augmentation_manager = AugmentationManager() if use_augmentation else None
    collected_reviews = []
    game_stats = {}
    global_counts_before = defaultdict(int)
    global_counts_after = defaultdict(int)
    total_augmented = 0
    total_subsampled = 0

    for gid in tqdm(all_game_ids, desc="Processing games", disable=not verbose):
        reviews_raw = merge_reviews_func(gid, source=source) or []
        if not reviews_raw:
            continue

        cat_counts_before = defaultdict(int)
        for r in reviews_raw:
            cat = getattr(r, 'category', None)
            if cat in categories:
                cat_counts_before[cat] += 1
                global_counts_before[cat] += 1

        if verbose:
            total = sum(cat_counts_before.values())
            print(f"\nGame {gid}: {total} reviews | " + " | ".join([f"{c}:{cat_counts_before[c]}" for c in categories]))

        balanced_reviews, stats = balance_single_game(
            reviews=reviews_raw,
            categories=categories,
            min_samples_for_balance=min_samples_for_balance,
            balance_strategy=balance_strategy,
            target_ratio=target_ratio,
            augmentation_manager=augmentation_manager,
            max_augmentations_per_review=max_augmentations_per_review,
            verbose=verbose
        )

        augmented_in_game = sum(1 for r in balanced_reviews if getattr(r, 'is_augmented', False))
        for r in balanced_reviews:
            cat = getattr(r, 'category', None)
            if cat in categories:
                global_counts_after[cat] += 1

        game_stats[gid] = {
            'counts_before': dict(cat_counts_before),
            'counts_after': {c: sum(1 for r in balanced_reviews if getattr(r, 'category', None) == c) for c in categories},
            'total_before': sum(cat_counts_before.values()),
            'total_after': len(balanced_reviews),
            'augmented_count': augmented_in_game,
            'subsampled_count': stats.get('subsampled_count', 0),
            'balance_stats': stats
        }

        total_augmented += augmented_in_game
        total_subsampled += stats.get('subsampled_count', 0)
        collected_reviews.extend(balanced_reviews)

    total_before = sum(global_counts_before.values())
    total_after = len(collected_reviews)
    ratio = max(global_counts_after.values()) / max(min(global_counts_after.values()), 1) if global_counts_after else 0

    if verbose:
        print(f"\n{'='*70}\nGLOBAL SUMMARY:")
        print(f"  Games processed: {len(game_stats)}/{len(all_game_ids)}")
        print(f"  Total reviews: {total_before} → {total_after}")
        print("  Before:  " + " | ".join([f"{c}:{global_counts_before[c]}" for c in categories]))
        print("  After:   " + " | ".join([f"{c}:{global_counts_after[c]}" for c in categories]))
        print(f"  Augmented: {total_augmented} ({(total_augmented/total_after*100 if total_after else 0):.1f}%)")
        print(f"  Subsampled: {total_subsampled}")
        print(f"  Final max/min ratio: {ratio:.2f}:1")
        print(f"  Strategy: {balance_strategy}\n{'='*70}\n")

    return collected_reviews, {
        'counts_before': dict(global_counts_before),
        'counts_after': dict(global_counts_after),
        'total_before': total_before,
        'total_after': total_after,
        'total_augmented': total_augmented,
        'total_subsampled': total_subsampled,
        'balance_ratio_max_min': ratio,
        'balance_strategy': balance_strategy,
        'game_stats': game_stats
    }