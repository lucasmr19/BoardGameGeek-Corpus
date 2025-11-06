from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict

from ..models import GameCorpus, Corpus, Review
from .metadata_utils import build_metadata
from .review_utils import merge_reviews
from .processing_utils import process_single_review
from ..preprocessing import init_spacy_models, filter_valid_reviews
from ..balancing import collect_balanced_reviews_multi_game, save_balance_report
from ..resources import LOGGER
from ..config import API_DIR, CRAWLER_DIR

def build_corpus(
    game_ids: List[int],
    source: str = "combined",
    data_api_dir: str = API_DIR,
    data_crawler_dir: str = CRAWLER_DIR,
    balance_strategy: str ='hybrid',
    min_samples_for_balance: int =30,
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
    Build the complete BGG corpus by merging, preprocessing into GameCorpus,
    filtering, and balancing reviews.
    """
    preprocess_kwargs = preprocess_kwargs or {}

    print("===============================================")
    print("PHASE 1: Merge and Preprocess Reviews into GameCorpus")
    print(f"Parallel: {parallel} ({max_workers} workers)")
    print("===============================================")

    # Step 1: Merge reviews per game
    merged_reviews_by_game = defaultdict(list)
    for gid in tqdm(game_ids, desc="Merging reviews per game", disable=not verbose):
        reviews = merge_reviews(
            game_id=gid,
            source=source,
            data_api_dir=data_api_dir,
            data_crawler_dir=data_crawler_dir
        )
        merged_reviews_by_game[gid].extend(reviews)

    # Step 2: Preprocess reviews into GameCorpus objects
    games = []
    if parallel:
        # single global pool + model preloading per worker
        with ProcessPoolExecutor(max_workers=max_workers, initializer=init_spacy_models) as executor:
            futures = {}
            for gid, reviews in tqdm(merged_reviews_by_game.items(), desc="Scheduling reviews"):
                meta = build_metadata(gid)
                game_corpus = GameCorpus(game_id=gid, metadata=meta, documents=[])

                for rev in reviews:
                    futures[executor.submit(process_single_review, rev, **preprocess_kwargs)] = (gid, game_corpus)

            # Collect results
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing all reviews"):
                gid, game_corpus = futures[f]
                try:
                    doc = f.result()
                    if doc:
                        doc.review.game_id = gid
                        game_corpus.add_document(doc)
                except Exception as e:
                    review_text = (getattr(rev, "comment")).strip()
                    preview = (review_text[:200] + "...") if len(review_text) > 200 else review_text
                    LOGGER.error(
                        f"[CorpusBuild] Error processing review in game {gid}: {e}\n"
                        f"→ Review preview: '{preview}'\n"
                        f"→ Username: {getattr(rev, 'username')} | Rating: {getattr(rev, 'rating')}"
                        f"| Timestamp: {getattr(rev, 'timestamp')}"
                    )


            # Consolidate unique GameCorpus instances
            game_map = {}
            for _, gc in futures.values():
                game_map[gc.game_id] = gc
            games = list(game_map.values())

    else:
        # Sequential mode (no multiprocessing, no model reloads)
        init_spacy_models()  # pre-load spaCy model(s) once in main process

        for gid, reviews in tqdm(merged_reviews_by_game.items(), desc="Processing games sequentially"):
            meta = build_metadata(gid)
            game_corpus = GameCorpus(game_id=gid, metadata=meta, documents=[])

            for rev in tqdm(reviews, desc=f"Processing reviews for game {gid}", leave=False):
                try:
                    doc = process_single_review(rev, **preprocess_kwargs)
                    doc.review.game_id = gid
                    game_corpus.add_document(doc)
                except Exception as e:
                    print(f"Error processing review in game {gid}: {e}")

            games.append(game_corpus)

    print("PHASE 2: Filter Reviews")
    games: List[GameCorpus] = filter_valid_reviews(games, min_tokens=2)

    print("PHASE 3: Balance Reviews Across Games")

    games, stats = collect_balanced_reviews_multi_game(
        games=games,
        categories=('positive', 'neutral', 'negative'),
        min_samples_for_balance=min_samples_for_balance,
        balance_strategy=balance_strategy,
        target_ratio=target_ratio,
        use_augmentation=use_augmentation,
        max_augmentations_per_review=max_augmentations_per_review,
        verbose=verbose
    )

    if save_report:
        save_balance_report(stats)

    total_docs = sum(len(g.documents) for g in games)
    print(f"\nCorpus successfully built. Games processed: {len(games)} | Total documents: {total_docs}")

    return Corpus(games=games), stats
