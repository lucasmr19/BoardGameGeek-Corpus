import os
from ..models import Review
from ..preprocessing import normalize_text
from .io_utils import load_json
from ..resources import DATA_API_DIR, DATA_CRAWLER_DIR


def merge_reviews(game_id, source="combined", data_api_dir=DATA_API_DIR, data_crawler_dir=DATA_CRAWLER_DIR):
    """
    Load and merge user reviews for a given game from both API and crawler sources.

    Depending on the `source` parameter, this function behaves as follows:
        - source="crawler": returns only crawler reviews.
        - source="api": returns only API reviews.
        - source="combined": merges both sources, avoiding duplicates using a normalized key
          based on (username + normalized comment).

    The merging strategy:
        1. Crawler reviews are added first (to preserve timestamps).
        2. API reviews are merged:
            - If the same (user, normalized comment) pair exists, the API review may replace
              the comment if it includes emoji shortcodes (":emoji:").
            - If not, the review is added as new.

    Args:
        game_id (str or int): The BGG game ID.
        source (str): One of {"crawler", "api", "combined"}. Default is "combined".
        data_api_dir (str): Directory containing API review files.
        data_crawler_dir (str): Directory containing crawler review files.

    Returns:
        list[Review]: A list of `Review` objects merged according to the source type.
    """
    api_file = os.path.join(data_api_dir, f"bgg_reviews_{game_id}_api.json")
    crawler_file = os.path.join(data_crawler_dir, f"bgg_reviews_{game_id}_crawler.json")

    data_api = load_json(api_file)
    data_crawler = load_json(crawler_file)

    # Extract API reviews list
    reviews_api = data_api.get("comments", []) if isinstance(data_api, dict) else data_api or []

    # ----------------------------------------
    # Case 1: Only Crawler
    # ----------------------------------------
    if source == "crawler":
        return [
            Review(
                username=r.get("username", "unknown"),
                rating=r.get("rating"),
                comment=r.get("comment", ""),
                timestamp=r.get("timestamp"),
                game_id=game_id
            )
            for r in (data_crawler or [])
        ]

    # ----------------------------------------
    # Case 2: Only API
    # ----------------------------------------
    if source == "api":
        return [
            Review(
                username=r.get("username", "unknown"),
                rating=r.get("rating"),
                comment=r.get("comment", ""),
                timestamp=None,
                game_id=game_id
            )
            for r in reviews_api
            if r.get("rating") not in (None, "N/A")
        ]

    # ----------------------------------------
    # Case 3: Combined (requires key normalization)
    # ----------------------------------------
    merged = {}

    def normalized_key(username, comment):
        """Create normalized merge key using username + normalized comment."""
        norm_comment = normalize_text(comment, lower=True, correct_spelling=False)
        return (username.strip().lower(), norm_comment)

    # Load crawler reviews first
    for r in (data_crawler or []):
        user = r.get("username", "unknown")
        comment = r.get("comment", "") or ""
        key = normalized_key(user, comment)
        merged[key] = Review(
            username=user,
            rating=r.get("rating"),
            comment=comment,
            timestamp=r.get("timestamp"),
            game_id=game_id
        )

    # Merge API reviews
    for r in reviews_api:
        user = r.get("username", "unknown")
        comment = r.get("comment", "") or ""
        rating_api = r.get("rating")

        if rating_api in (None, "N/A"):
            continue

        key = normalized_key(user, comment)

        if key in merged:
            existing = merged[key]
            # Prefer API comment if it includes emoji shortcodes
            if r.get("comment") and ":" in r["comment"]:
                existing.comment = r["comment"]
        else:
            merged[key] = Review(
                username=user,
                rating=rating_api,
                comment=comment,
                timestamp=None,
                game_id=game_id
            )

    return list(merged.values())