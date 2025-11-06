from collections import defaultdict
from tqdm import tqdm
from typing import List, Dict, Tuple

from .augmentation import AugmentationManager
from .single_game_balance import balance_single_game
from ..models import GameCorpus


def collect_balanced_reviews_multi_game(
    games: List[GameCorpus],
    categories=('positive', 'neutral', 'negative'),
    min_samples_for_balance=30,
    balance_strategy='hybrid',
    target_ratio=0.6,
    use_augmentation=True,
    max_augmentations_per_review=2,
    verbose=True
) -> Tuple[List[GameCorpus], Dict]:
    """
    Balance reviews across multiple GameCorpus objects, returning a new list of GameCorpus
    with balanced CorpusDocuments per game.
    """

    augmentation_manager = AugmentationManager() if use_augmentation else None
    balanced_games = []
    game_stats = {}
    global_counts_before = defaultdict(int)
    global_counts_after = defaultdict(int)
    total_augmented = 0
    total_subsampled = 0

    for game in tqdm(games, desc="Balancing reviews per game", disable=not verbose):
        docs = game.documents
        if not docs:
            continue

        cat_counts_before = defaultdict(int)
        for d in docs:
            cat = getattr(d.review, "category", None)
            if cat in categories:
                cat_counts_before[cat] += 1
                global_counts_before[cat] += 1

        balanced_docs, stats = balance_single_game(
            documents=docs,
            categories=categories,
            min_samples_for_balance=min_samples_for_balance,
            balance_strategy=balance_strategy,
            target_ratio=target_ratio,
            augmentation_manager=augmentation_manager,
            max_augmentations_per_review=max_augmentations_per_review,
            verbose=verbose,
        )

        new_game = GameCorpus(game_id=game.game_id, metadata=game.metadata, documents=balanced_docs)
        balanced_games.append(new_game)

        augmented_in_game = sum(1 for d in balanced_docs if getattr(d.review, "is_augmented", False))
        for d in balanced_docs:
            cat = getattr(d.review, "category", None)
            if cat in categories:
                global_counts_after[cat] += 1

        game_stats[game.game_id] = {
            "counts_before": dict(cat_counts_before),
            "counts_after": {c: sum(1 for d in balanced_docs if getattr(d.review, "category") == c) for c in categories},
            "total_before": sum(cat_counts_before.values()),
            "total_after": len(balanced_docs),
            "augmented_count": augmented_in_game,
            "subsampled_count": stats.get("subsampled_count", 0),
            "balance_stats": stats,
        }

        total_augmented += augmented_in_game
        total_subsampled += stats.get("subsampled_count", 0)

    # 🔹 Aggregate global stats
    total_before = sum(global_counts_before.values())
    total_after = sum(global_counts_after.values())
    ratio = (
        max(global_counts_after.values()) / max(min(global_counts_after.values()), 1)
        if global_counts_after else 0
    )

    if verbose:
        print(f"\n{'=' * 70}\nGLOBAL SUMMARY:")
        print(f"  Games processed: {len(game_stats)}")
        print(f"  Total reviews: {total_before} → {total_after}")
        print("  Before:  " + " | ".join([f"{c}:{global_counts_before[c]}" for c in categories]))
        print("  After:   " + " | ".join([f"{c}:{global_counts_after[c]}" for c in categories]))
        print(f"  Augmented: {total_augmented} ({(total_augmented / max(total_after,1) * 100):.1f}%)")
        print(f"  Subsampled: {total_subsampled}")
        print(f"  Final max/min ratio: {ratio:.2f}:1")
        print(f"  Strategy: {balance_strategy}\n{'=' * 70}\n")

    return balanced_games, {
        "counts_before": dict(global_counts_before),
        "counts_after": dict(global_counts_after),
        "total_before": total_before,
        "total_after": total_after,
        "total_augmented": total_augmented,
        "total_subsampled": total_subsampled,
        "balance_ratio_max_min": ratio,
        "balance_strategy": balance_strategy,
        "game_stats": game_stats,
    }
