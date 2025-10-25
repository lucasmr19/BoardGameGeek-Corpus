import os
import pandas as pd
from .io_utils import load_json
from ..resources import DATA_API_DIR
from ..config import RANKS_DF, BGG_STATS_DF

def load_metadata_from_api(game_id, data_api_dir=DATA_API_DIR):
    meta_file = os.path.join(data_api_dir, f"bgg_metadata_{game_id}_api.json")
    return load_json(meta_file) or {}

def build_metadata(game_id, ranks_df=RANKS_DF, stats_df=BGG_STATS_DF, data_api_dir=DATA_API_DIR):
    """Build structured metadata for a game by aggregating data from multiple sources."""
    game_metadata = load_metadata_from_api(game_id, data_api_dir)
    
    # ranks (from BoardGeekGames-Corpus/data/raw/boardgames_ranks.csv)
    rank_row = ranks_df[ranks_df["id"] == game_id] if not ranks_df.empty else pd.DataFrame()
    rank_data = rank_row.iloc[0].to_dict() if not rank_row.empty else {}
    rank_data = {k: (None if pd.isna(v) else v) for k, v in rank_data.items()}

    # stats (from bgg_stats.csv)
    stats_row = stats_df[stats_df["game_id"] == game_id] if not stats_df.empty else pd.DataFrame()
    stats_data = stats_row.iloc[0].to_dict() if not stats_row.empty else {}
    stats_data = {k: (None if pd.isna(v) else v) for k, v in stats_data.items()}

    meta_raw = {**(game_metadata or {}), **rank_data, **stats_data}
    classifications_raw = meta_raw.get("classifications", {})

    metadata = {
        "game_info": {
            #"id": meta_raw.get("id", game_id),
            "name": meta_raw.get("name"),
            "yearpublished": meta_raw.get("yearpublished"),
            "minplayers": meta_raw.get("minplayers"),
            "maxplayers": meta_raw.get("maxplayers"),
            "minplaytime": meta_raw.get("minplaytime"),
            "maxplaytime": meta_raw.get("maxplaytime"),
            "age": meta_raw.get("age"),
            "description": meta_raw.get("description"),
            "image": meta_raw.get("image"),
            "thumbnail": meta_raw.get("thumbnail"),
            "designers": meta_raw.get("designers", []),
            "artists": meta_raw.get("artists", []),
            "publishers": meta_raw.get("publishers", []),
            "is_expansion": bool(meta_raw.get("is_expansion", 0))
        },
        "stats": {
            "num_reviews": meta_raw.get("total_all"),
            "num_reviews_commented": meta_raw.get("total_commented"),
            "num_reviews_rated": meta_raw.get("total_rated"),
            "num_reviews_rated_and_commented": meta_raw.get("total_rated_and_commented"),
            "avg_rating": meta_raw.get("average"),
            "bayes_average": meta_raw.get("bayesaverage"),
            "avg_weight": meta_raw.get("avgweight"),
            "num_weights": meta_raw.get("numweights")
        },
        "rankings": {
            "overall_rank": meta_raw.get("rank"),
            "strategygames_rank": meta_raw.get("strategygames_rank"),
            "thematic_rank": meta_raw.get("thematic_rank"),
            "familygames_rank": meta_raw.get("familygames_rank"),
            "partygames_rank": meta_raw.get("partygames_rank"),
            "cgs_rank": meta_raw.get("cgs_rank"),
            "childrensgames_rank": meta_raw.get("childrensgames_rank"),
            "abstracts_rank": meta_raw.get("abstracts_rank"),
            "wargames_rank": meta_raw.get("wargames_rank")
        },
        "classifications": {
            "mechanics": classifications_raw.get("mechanics", []),
            "categories": classifications_raw.get("categories", []),
            "families": classifications_raw.get("families", [])
        },
        "polls": {
            "complexity_poll": {
                "poll_avg": meta_raw.get("poll_avg"),
                "poll_votes": meta_raw.get("poll_votes")
            }
        }
    }
    return metadata
