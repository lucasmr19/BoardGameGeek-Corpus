#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
pln_p3_7462_02_e3.py

Opinion mining with LLM prompting (English)
-------------------------------------------------------
This script prepares detailed, structured prompts for two tasks:

  - Task 3a: Aggregate opinion summarization (returns JSON with summary, positives,
             negatives, conflicting opinions and top aspects).
  - Task 3b: Aspect-based opinion extraction (returns JSON listing aspects,
             inferred sentiment and supporting evidence snippets).

Features:
  - Dynamically reads the game's name from the corpus using --game_id.
  - Extracts "clean_text" review fields for the chosen game.
  - Allows limiting/sampling reviews to control prompt size (--max_reviews).
  - Performs simple chunking or sampling if reviews are too large for a single prompt.
  - Always instructs the model to return valid JSON only (for copy/paste to local files).
  - Optionally saves the generated prompt to a file.

Usage examples:
  python pln_p3_7462_02_e3.py --game_id 1 --task 3a --corpus_path data/processed/corpora/bgg_corpus.json
  python pln_p3_7462_02_e3.py --game_id 1 --task 3b --max_reviews 200 --save_prompt prompt_3b.json
"""

from __future__ import annotations

import os
import json
import argparse
import random
from typing import List, Optional
from collections import defaultdict

from bgg_corpus.resources import LOGGER
from src.bgg_corpus.config import CORPORA_DIR, CORPUS_NAME


# =====================================================
# Corpus loading
# =====================================================
def load_corpus(corpus_path: str) -> List[dict]:
    """Load the corpus JSON file and return the list of game objects."""
    if not os.path.isfile(corpus_path):
        LOGGER.error("Corpus path does not exist: %s", corpus_path)
        raise FileNotFoundError(corpus_path)

    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = json.load(f)
    return corpus


def load_game_by_id(corpus_path: str, game_id: int) -> Optional[dict]:
    """Return the game dict for the given game_id, or None if not found."""
    corpus = load_corpus(corpus_path)
    for game in corpus:
        if game.get("game_id") == game_id or str(game.get("game_id")) == str(game_id):
            return game
    return None


def extract_clean_reviews(game_obj: dict) -> List[dict]:
    """
    Extract reviews as dictionaries containing:
      - text (clean_text)
      - category (negative | neutral | positive)
      - rating (float)
    """
    reviews = []
    for r in game_obj.get("reviews", []):
        txt = r.get("clean_text") or r.get("raw_text")
        if not txt:
            continue

        reviews.append({
            "text": " ".join(txt.split()),
            "category": r.get("category"),
            "rating": r.get("rating")
        })

    return reviews


# =====================================================
# BALANCED-BY-RATING SAMPLING STRATEGY
# =====================================================

RATING_BUCKETS = {
    "negative": [1, 2, 3, 4],        # 4 subgroups: 8.33% each
    "neutral": [5, 6],              # 2 subgroups: 16.67% each
    "positive": [7, 8, 9, 10]       # 4 subgroups: 8.33% each
}


def sample_reviews_balanced_by_rating(
    reviews: List[dict],
    max_reviews: int,
    seed: int = 42
) -> List[dict]:
    """
    Balanced sampling preserving the intrinsic rating distribution
    inside each sentiment class.

    Distribution reproduced:
        Negative: ratings 1-4 (4 equal subgroups)
        Neutral:  ratings 5-6 (2 equal subgroups)
        Positive: ratings 7-10 (4 equal subgroups)

    Steps:
      1. Split reviews by (category → rating).
      2. Allocate exactly 1/3 of max_reviews to each category.
      3. Within each category, allocate reviews evenly across rating buckets.
      4. If any subgroup does not have enough reviews, fill deficit from the
         remaining pool (best-effort).

    Returns:
        A list of review dictionaries.
    """
    random.seed(seed)

    # 1. Group reviews by category → rating
    grouped = defaultdict(lambda: defaultdict(list))
    for r in reviews:
        cat = r.get("category")
        rating = int(r.get("rating"))
        grouped[cat][rating].append(r)

    # 2. Target number of reviews per category
    target_per_class = max_reviews // 3
    selected = []

    # 3. Sample respecting rating-level balance
    for category, ratings in RATING_BUCKETS.items():
        n_subgroups = len(ratings)
        target_per_rating = target_per_class // n_subgroups

        for rt in ratings:
            candidates = grouped[category][rt]
            n_take = min(len(candidates), target_per_rating)

            if n_take > 0:
                selected.extend(random.sample(candidates, n_take))

    # 4. Fill any shortage
    if len(selected) < max_reviews:
        remaining = [r for r in reviews if r not in selected]
        random.shuffle(remaining)
        selected.extend(remaining[: max_reviews - len(selected)])

    return selected


# =====================================================
# Standard sampling strategies
# =====================================================
def sample_reviews(
    reviews: List[dict],
    max_reviews: Optional[int],
    strategy: str = "latest",
    seed: int = 42
) -> List[dict]:
    """
    Select a subset of reviews according to sampling strategy.
    Supports:
        - latest:   first N reviews
        - random:   shuffled selection
        - longest:  sort by length and take N
        - balanced_by_rating: reproduce intrinsic class-rating distribution
    """
    if max_reviews is None or max_reviews >= len(reviews):
        return reviews

    if strategy == "balanced_by_rating":
        return sample_reviews_balanced_by_rating(reviews, max_reviews, seed=seed)

    if strategy == "latest":
        return reviews[:max_reviews]

    if strategy == "random":
        rnd = random.Random(seed)
        rnd.shuffle(reviews)
        return reviews[:max_reviews]

    if strategy == "longest":
        return sorted(reviews, key=lambda r: len(r["text"]), reverse=True)[:max_reviews]

    # fallback
    return reviews[:max_reviews]


# =====================================================
# Chunking helper
# =====================================================
def chunk_reviews_by_char_limit(reviews: List[dict], char_limit: int) -> List[str]:
    """
    Create text chunks under a maximum character limit.
    Returns a list of raw text blocks (joined with double newlines).
    """
    if char_limit is None or char_limit <= 0:
        return ["\n\n".join([r["text"] for r in reviews])]

    chunks = []
    current = []
    current_len = 0

    for r in reviews:
        t = r["text"]
        t_len = len(t) + 2
        if current_len + t_len > char_limit and current:
            chunks.append("\n\n".join(current))
            current = [t]
            current_len = t_len
        else:
            current.append(t)
            current_len += t_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks

# ------------------------------
# Prompt builders
# ------------------------------
def build_prompt_opinion_summary(game_name: str, reviews_text: str, n_reviews: int) -> str:
    """
    Build a detailed prompt for Task 3a (aggregate opinion summary).
    The LLM is required to output a single JSON object only — no other text.
    """
    schema_example = {
        "game_name": game_name,
        "n_reviews_analyzed": n_reviews,
        "summary": "<short neutral synthesized summary>",
        "top_aspects": [
            {"aspect": "<aspect name>", "sentiment": "positive|negative|neutral", "weight": "qualitative or numeric importance", "evidence": ["snippet1", "snippet2"]}
        ],
        "positives": ["short aspect-level bullet strings"],
        "negatives": ["short aspect-level bullet strings"],
        "conflicting_opinions": ["short sentences describing disagreements or trade-offs"]
    }

    prompt = f"""
        You are an expert in opinion mining and summarization of user reviews.
        Task: produce an aggregated, factual and neutral opinion summary for the board game "{game_name}".

        REQUIREMENTS (READ CAREFULLY):
        1) Your response MUST be valid JSON and ONLY a single JSON object (no markdown, no code fences, no surrounding commentary).
        2) Do not invent facts: every claim must be grounded in the provided review text.
        3) If you cannot find evidence for an item, set it to an empty list or an empty string as appropriate.
        4) Minimize redundancy: keep summary concise (max ~3-5 sentences).

        OUTPUT SCHEMA (the JSON object must follow this schema exactly):
        {json.dumps(schema_example, indent=2)}

        FIELDS DEFINITION:
        - game_name: the game's name (string).
        - n_reviews_analyzed: integer number of reviews used.
        - summary: 2-5 sentence neutral synthesis describing overall opinion and main themes.
        - top_aspects: list of aspect objects (aspect name, sentiment, a short 'weight' indicating importance, and 1-3 evidence snippets).
        - positives: short list of main positive aspects (strings).
        - negatives: short list of main negative aspects (strings).
        - conflicting_opinions: list of short descriptions of disagreements between reviewers (e.g., 'Some find it too long; others praise the depth').

        REVIEWS (do not invent anything beyond these reviews):
        {reviews_text}

        IMPORTANT: Return only JSON. End of task.
        """.strip()
    return prompt


def build_prompt_aspects_extraction(game_name: str, reviews_text: str, n_reviews: int) -> str:
    """
    Build a detailed prompt for Task 3b (aspect extraction).
    The LLM is required to output valid JSON only.
    """
    schema_example = {
        "game_name": game_name,
        "n_reviews_analyzed": n_reviews,
        "aspects": [
            {
                "aspect": "<name>",
                "category": "mechanics|components|theme|rules|player_interaction|playtime|other",
                "sentiment": "positive|negative|neutral",
                "evidence": ["short supporting snippet 1", "snippet 2"],
                "examples_count": 0
            }
        ]
    }

    prompt = f"""
        You are an expert in aspect-based sentiment analysis.

        Task: extract structured opinions from user reviews about the board game "{game_name}".

        REQUIREMENTS:
        1) Return ONLY a valid JSON object that matches the schema below (no extra explanatory text).
        2) For each aspect included, supply:
        - aspect: concise aspect name (string).
        - category: one of {["mechanics","components","theme","rules","player_interaction","playtime","other"]}.
        - sentiment: one of "positive", "negative", or "neutral".
        - evidence: 1-3 short snippets from the reviews that justify the sentiment.
        - examples_count: an integer approximating how many reviews mention this aspect (best-effort).
        3) Do NOT invent aspects or sentiments not supported by the review text.
        4) When multiple synonyms exist, group them under a single aspect name (e.g., "playtime" and "duration").
        5) If an aspect appears but has mixed sentiment, set sentiment to "neutral" and include evidence for both sides in the evidence list.

        OUTPUT SCHEMA (example — your JSON must follow this structure exactly):
        {json.dumps(schema_example, indent=2)}

        REVIEWS (use only these to infer aspects and sentiments):
        {reviews_text}

        IMPORTANT: Return only JSON. End of task.
        """.strip()
    return prompt


# =====================================================
# Main CLI
# =====================================================
def main():
    parser = argparse.ArgumentParser(description="Generate structured LLM prompts for BGG opinion mining (tasks 3a / 3b).")
    parser.add_argument("--game_id", type=int, required=True)
    parser.add_argument("--task", type=str, choices=["3a", "3b"], required=True)
    parser.add_argument("--corpus_path", type=str, default=os.path.join(CORPORA_DIR, f"{CORPUS_NAME}.json"))
    parser.add_argument("--max_reviews", type=int, default=500)
    parser.add_argument(
        "--sample_strategy",
        type=str,
        choices=["latest", "random", "longest", "balanced_by_rating"],
        default="latest",
        help="Sampling method."
    )
    parser.add_argument("--char_limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_prompt", type=str, default=None)
    parser.add_argument("--print_prompt", action="store_true")
    args = parser.parse_args()

    LOGGER.info("Loading game with id=%s from corpus=%s", args.game_id, args.corpus_path)
    game = load_game_by_id(args.corpus_path, args.game_id)
    if game is None:
        LOGGER.error("Game id %s not found", args.game_id)
        raise SystemExit(1)

    try:
        game_name = game["metadata"]["game_info"]["name"]
    except Exception:
        LOGGER.warning("Could not extract game name; using fallback.")
        game_name = game.get("name") or f"game_{args.game_id}"

    reviews = extract_clean_reviews(game)
    if not reviews:
        LOGGER.warning("No reviews found.")
        raise SystemExit(0)

    # Apply sampling strategy
    sampled = sample_reviews(
        reviews,
        args.max_reviews,
        strategy=args.sample_strategy,
        seed=args.seed
    )

    # Chunking logic
    if args.char_limit and args.char_limit > 0:
        chunks = chunk_reviews_by_char_limit(sampled, args.char_limit)
        chosen_chunk = chunks[0]
        n_reviews_used = len(chosen_chunk.split("\n\n"))
    else:
        chosen_chunk = "\n\n".join(r["text"] for r in sampled)
        n_reviews_used = len(sampled)

    # Build prompt
    if args.task == "3a":
        prompt = build_prompt_opinion_summary(game_name, chosen_chunk, n_reviews_used)
    else:
        prompt = build_prompt_aspects_extraction(game_name, chosen_chunk, n_reviews_used)

    # Output handling
    if args.save_prompt:
        os.makedirs(os.path.dirname(args.save_prompt) or ".", exist_ok=True)
        with open(args.save_prompt, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "game_id": args.game_id,
                    "game_name": game_name,
                    "task": args.task,
                    "n_reviews_used": n_reviews_used
                },
                "prompt": prompt
            }, f, indent=2, ensure_ascii=False)
        LOGGER.info("Prompt saved to %s", args.save_prompt)

    if args.print_prompt or not args.save_prompt:
        print("\n" + "=" * 80)
        print(f"GENERATED PROMPT (TASK {args.task}) — Game: {game_name} — Reviews used: {n_reviews_used}")
        print("=" * 80 + "\n")
        print(prompt)
        print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()