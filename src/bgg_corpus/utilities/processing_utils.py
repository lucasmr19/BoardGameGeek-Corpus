from ..models import Review, CorpusDocument
from ..preprocessing import process_review_item

def process_single_review(review_item: Review, **preprocess_kwargs):
    """
    Process a Review and transform the result into a CorpusDocument object.
    Args:
      - dict con keys 'username','rating','raw_text'/'comment','timestamp','game_id', 'clean_text'
    Returns a CorpusDocument or None.
    """
    rev = review_item

    processed = None
    # submit the item to the process_review_item pipeline
    processed = process_review_item(
        item={
            "username": rev.username,
            "rating": rev.rating,
            "timestamp": rev.timestamp,
            "raw_text": rev.comment,
            "clean_text": rev.clean_text,
            "game_id": rev.game_id,
        },
        **preprocess_kwargs
    )
    # consistency
    processed.setdefault("label", getattr(rev, "label", None))
    return CorpusDocument(rev, processed) if processed else None