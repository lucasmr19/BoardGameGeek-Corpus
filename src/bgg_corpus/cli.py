#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import Counter
from bgg_corpus.utilities import build_corpus, generate_corpus_statistics
from bgg_corpus.storage import MongoCorpusStorage
from bgg_corpus.config import CORPORA_DIR, CORPUS_NAME, CORPORA_STATISTICS_DIR, RANKS_DF
from bgg_corpus.resources import LOGGER

# ----------------------------
# Default configuration
# ----------------------------
DEFAULT_OUTPUT_DIR = CORPORA_DIR
DEFAULT_OUTPUT_JSON_NAME = f"{CORPUS_NAME}.json"
DEFAULT_MAX_WORKERS = 4
all_ids = RANKS_DF['id'].tolist()
DEFAULT_GAMES = list(range(1, 510)) + all_ids[:100] # All the reviews downloaded in crawler/API


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BoardGameGeek corpus and generate statistics.")

    # Game selection
    parser.add_argument("--games", nargs="+", type=int, default=DEFAULT_GAMES, help="List of game_ids to process (e.g., --games 2 224517).")

    # Output
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save the JSON output (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_JSON_NAME, help=f"Output JSON filename (default: {DEFAULT_OUTPUT_JSON_NAME}).")
    parser.add_argument("--save-json", action="store_true", help="Save corpus as JSON file.")
    parser.add_argument("--save-mongo", action="store_true", help="Save corpus to MongoDB.")

    # Parallelism
    parser.add_argument("--no-parallel", action="store_true", help="Disable parallel processing.")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"Maximum workers for parallel processing (default: {DEFAULT_MAX_WORKERS}).")

    # Build corpus options
    parser.add_argument("--source", type=str, default="crawler", choices=["crawler", "api", "combined"], help="Source of reviews.")
    parser.add_argument("--balance-strategy", type=str, default="undersample", choices=["oversample", "undersample", "hybrid"], help="Strategy to balance reviews.")
    parser.add_argument("--min-samples", type=int, default=30, help="Minimum samples for balancing.")
    parser.add_argument("--target-ratio", type=float, default=None, help="Target ratio for hybrid balance strategy.")
    parser.add_argument("--use-augmentation", action="store_true", help="Use text augmentation for balancing.")
    parser.add_argument("--max-augmentations-per-review", type=int, default=2, help="Max augmentations per review.")
    parser.add_argument("--save-report", action="store_true", help="Save balance report as JSON.")

    # Post-build the corpus
    parser.add_argument("--generate-stats", action="store_true", help="Generate and display corpus statistics after building.")

    args = parser.parse_args()

    # ----------------------------
    # Determine game IDs
    # ----------------------------
    game_ids = args.games
    if not game_ids or game_ids == [0]:
        raise SystemExit("Please provide --games variable.")

    LOGGER.info(f"Processing game_ids: {game_ids}")

    # ----------------------------
    # Ensure output directory
    # ----------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_name)

    # ----------------------------
    # Build corpus
    # ----------------------------
    corpus, stats = build_corpus(
        game_ids=game_ids,
        source=args.source,
        balance_strategy=args.balance_strategy,
        min_samples_for_balance=args.min_samples,
        target_ratio=args.target_ratio,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        use_augmentation=args.use_augmentation,
        max_augmentations_per_review=args.max_augmentations_per_review,
        save_report=args.save_report,
        verbose=True
    )

    # ----------------------------
    # Save corpus
    # ----------------------------
    # Save to json
    if args.save_json:
        corpus.to_json(output_path)
        LOGGER.info(f"Corpus built with {len(corpus.documents)} reviews. Saved at {output_path}")

    # Save to MongoDB
    if args.save_mongo:
        mongo_storage = MongoCorpusStorage()
        corpus.save_to_mongo(mongo_storage)
        LOGGER.info(f"Corpus saved to MongoDB with {len(corpus.documents)} documents."
                    f"MongoDB info: {mongo_storage}")

    # ----------------------------
    # General statistics
    # ----------------------------
    if args.generate_stats:
        generate_corpus_statistics(corpus, base_path=CORPORA_STATISTICS_DIR)