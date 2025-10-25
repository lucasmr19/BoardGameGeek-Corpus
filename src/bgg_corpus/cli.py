#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from collections import Counter
from bgg_corpus.utilities import build_corpus
from bgg_corpus.storage import MongoCorpusStorage
from bgg_corpus.config import CORPORA_DIR
from bgg_corpus.resources import LOGGER

# ----------------------------
# Default configuration
# ----------------------------
DEFAULT_OUTPUT_DIR = CORPORA_DIR
DEFAULT_OUTPUT_NAME = "bgg_corpus.json"
DEFAULT_MAX_WORKERS = 4
DEFAULT_GAMES = [3]


# ----------------------------
# Main execution
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build BoardGameGeek corpus and generate statistics.")

    # Game selection
    parser.add_argument("--games", nargs="+", type=int, default=DEFAULT_GAMES, help="List of game_ids to process (e.g., --games 2 224517).")

    # Output
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help=f"Directory to save the JSON output (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--output-name", type=str, default=DEFAULT_OUTPUT_NAME, help=f"Output JSON filename (default: {DEFAULT_OUTPUT_NAME}).")
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
        print("\n📊 General statistics")
        print("- Total reviews:", corpus.num_reviews())
        print("- Rated reviews:", corpus.num_reviews_rated())
        print("- Reviews with text:", corpus.num_reviews_commented())
        print("- Reviews with rating & text:", corpus.num_reviews_rated_and_commented())
        print("- Unique users:", corpus.num_unique_users())
        print("- Non-unique users:", corpus.num_no_unique_users())
        print("- 10 non-unique users:", corpus.no_unique_users()[:10])

        # Ratings distribution
        print("\n⭐ Ratings distribution (top 10)")
        rating_dist = corpus.rating_distribution()
        for rating, count in sorted(rating_dist.items(), key=lambda x: -x[1])[:10]:
            print(f"  {rating:>4}: {count}")

        # Sample raw text
        print("\n📝 Sample raw text:")
        print(corpus.raw()[:5], "...\n")

        # Word contexts
        print("🔍 Contexts for 'a' (window=3, first 5):")
        for ctx in corpus.contexts("a", window=3)[:5]:
            print(" ", ctx)

        print("\n⚖️ Top 5 common contexts for ['good', 'bad'] (window=2):")
        print(corpus.common_contexts(["good", "bad"], window=2)[:5])

        # Word frequency
        print("\n📌 Most frequent words (top 15):")
        for word, freq in corpus.most_common(15):
            print(f"  {word}: {freq}")

        # Lexical dispersion
        print("\n📈 Lexical dispersion: plotting graph...")
        corpus.lexical_dispersion_plot(["good", "game", "player"])

        # Hapax legomena
        print("\n🟢 Hapax legomena (first 20):")
        print(corpus.hapaxes()[:20])

        # Word length distribution
        print("\n📏 Word length distribution (top 10):")
        length_dist = corpus.word_length_distribution()
        for length, count in sorted(length_dist.items())[:10]:
            print(f"  Length {length}: {count} occurrences")

        print("\n📊 Plotting word length distribution...")
        corpus.plot_word_length_distribution()

        # N-grams and collocations
        print("\n🔗 Top 10 bigrams:")
        for bg, freq in Counter(corpus.bigrams()).most_common(10):
            print(f"  {bg}: {freq}")

        print("\n🔗 Top 5 trigrams:")
        for tg, freq in Counter(corpus.trigrams()).most_common(5):
            print(f"  {tg}: {freq}")

        print("\n📎 Top 10 collocations:")
        for coll, freq in corpus.collocations(10):
            print(f"  {coll}: {freq}")

        # Category comparison
        print("\n📂 Token counts by category:")
        corpus.print_category_stats()

        print("\n📂 Review counts by category:")
        corpus.print_review_counts()

        # Word frequency visualization
        print("\n📊 Word frequency visualization (top 30):")
        corpus.plot_frequency_distribution(30, title="Corpus Word Frequency")
