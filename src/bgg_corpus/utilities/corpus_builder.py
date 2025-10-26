from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from functools import partial

from ..models import GameCorpus, Corpus
from .metadata_utils import build_metadata
from .review_utils import merge_reviews
from .processing_utils import process_single_review
from ..balancing import collect_balanced_reviews_multi_game, save_balance_report

def ensure_review_obj(r, gid):
        """Ensure a standardized review object format (handles dict or Review)."""
        if isinstance(r, dict):
            out = dict(r)
            out["game_id"] = gid
            out.setdefault("raw_text", out.get("comment", ""))
            return out
        return {
            "username": getattr(r, "username", "unknown"),
            "rating": getattr(r, "rating", None),
            "comment": getattr(r, "comment", ""),
            "raw_text": getattr(r, "raw_text", ""),
            "timestamp": getattr(r, "timestamp", None),
            "game_id": gid
        }

def build_corpus(
    game_ids,
    source="combined",
    balance_strategy='hybrid',
    min_samples_for_balance=30,
    target_ratio=None,
    parallel=True,
    max_workers=4,
    use_augmentation=False,
    max_augmentations_per_review=2,
    save_report=True,
    verbose=True,
    preprocess_kwargs=None
):
    """
    Build the complete BGG corpus by combining review merging, multi-game balancing,
    and parallelized text processing.

    This function performs the following main stages:
        1. **Review collection and balancing per game** using the configured strategy
           (oversampling, undersampling, or hybrid). Balancing ensures inter/intra-class
           balance, including special handling for neutral (5-6 rating) reviews.
        2. **Parallel preprocessing** of all reviews into standardized `CorpusDocument` objects.
        3. **Corpus assembly** into `GameCorpus` and top-level `Corpus` objects.

    Args:
        game_ids (list[str] or list[int]): List of game IDs to include in the corpus.
        source (str): Review source mode. Options: {"api", "crawler", "combined"}.
                      Only uses merge key computation when `combined` for performance.
        balance_strategy (str): Strategy for balancing reviews across categories.
                                Options: {"oversample", "undersample", "hybrid"}.
        min_samples_for_balance (int): Minimum number of samples per category for balancing.
        target_ratio (float, optional): Ratio (minority/majority) used for hybrid balancing.
        parallel (bool): Whether to process reviews in parallel using multiprocessing.
        max_workers (int): Maximum number of parallel workers for processing.
        use_augmentation (bool): Whether to use text augmentation for underrepresented categories.
        max_augmentations_per_review (int): Max number of augmentations per review.
        save_report (bool): Whether to save the balancing report after phase 1.
        verbose (bool): Whether to show progress and detailed logs.

    Returns:
        Tuple[Corpus, dict]:
            - **Corpus**: The complete corpus containing all processed games and documents.
            - **dict**: Balancing statistics aggregated across all games.

    Workflow:
        1. Collect and balance reviews per game.
        2. Process each review into a standardized format.
        3. Build a structured corpus (`Corpus` -> `GameCorpus` -> `CorpusDocument`).

    Example:
        >>> corpus, stats = build_corpus(
        ...     game_ids=[12345, 67890],
        ...     balance_strategy='hybrid',
        ...     use_augmentation=True,
        ...     parallel=True
        ... )
        >>> print(len(corpus.games))
        2
        >>> print(stats['summary'])
    """
    preprocess_kwargs = preprocess_kwargs or {}
    print("===============================================")
    print("PHASE 1: Review Collection and Balancing per Game")
    print(f"Strategy: {balance_strategy}")
    print(f"Parallel: {parallel} ({max_workers} workers)")
    print("===============================================")

    # 1. Collect and balance reviews per game
    merge_func = partial(merge_reviews, source=source)

    collected_reviews, stats = collect_balanced_reviews_multi_game(
        all_game_ids=game_ids,
        merge_reviews_func=merge_func,
        min_samples_for_balance=min_samples_for_balance,
        balance_strategy=balance_strategy,
        target_ratio=target_ratio,
        use_augmentation=use_augmentation,
        max_augmentations_per_review=max_augmentations_per_review,
        source=source,
        verbose=verbose
    )

    if save_report:
        save_balance_report(stats)

    # 2. Convert balanced reviews to CorpusDocument
    print("\nPHASE 2: Converting to CorpusDocument")

    game_groups = defaultdict(list)
    for idx, r in enumerate(collected_reviews):
        gid = getattr(r, 'game_id', None)
        if gid:
            game_groups[gid].append(r)
        else:
            print(f"Review missing game_id: {r}")

    games = []

    # 3. Process each game's reviews (parallel or sequential)
    for gid, reviews in tqdm(game_groups.items(), desc="Creating GameCorpus"):
        meta = build_metadata(gid)
        game_corpus = GameCorpus(game_id=gid, metadata=meta, documents=[])

        if parallel and len(reviews) > 0:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for r in reviews:
                    rev_dict = ensure_review_obj(r, gid)
                    futures[executor.submit(process_single_review, rev_dict, **preprocess_kwargs)] = rev_dict

                for f in tqdm(as_completed(futures), total=len(futures),
                              desc=f"Processing reviews for game {gid}", leave=False):
                    try:
                        doc = f.result()
                        if doc:
                            doc.review.game_id = gid
                            doc.review.fileid = gid
                            doc.fileid = gid
                            game_corpus.add_document(doc)
                    except Exception as e:
                        print(f"Error processing review in game {gid}: {e}")
        else:
            for r in tqdm(reviews, desc=f"Processing reviews for game {gid}", leave=False):
                rev_dict = ensure_review_obj(r, gid)
                doc = process_single_review(rev_dict, **preprocess_kwargs)
                if doc:
                    doc.review.game_id = gid
                    doc.review.fileid = gid
                    doc.fileid = gid
                    game_corpus.add_document(doc)

        games.append(game_corpus)

    # 4. Final summary
    total_docs = sum(len(g.documents) for g in games)
    print("\nCorpus successfully built.")
    print(f"Games processed: {len(games)} | Total documents: {total_docs}")

    return Corpus(games=games), stats