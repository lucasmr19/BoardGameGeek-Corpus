from ..models import Review, CorpusDocument
from ..preprocessing import process_review_item

def process_single_review(review_item, **preprocess_kwargs):
    """
    Procesa un Review o un dict en un CorpusDocument.
    Acepta:
      - instancia de Review
      - dict con keys 'username','rating','raw_text'/'comment','timestamp','game_id'
    Devuelve CorpusDocument o None.
    """
    # normalizar input: reconstruir Review si viene dict
    if isinstance(review_item, dict):
        rev = Review.from_dict(review_item)
    else:
        rev = review_item

    processed = None
    if (rev.comment or "").strip():
        # aplicar pre_normalize antes de procesar para asegurar [thing] y html fixes
        processed = process_review_item(
            item={
                "username": rev.username,
                "rating": rev.rating,
                "timestamp": rev.timestamp,
                "raw_text": rev.comment
            },
            **preprocess_kwargs
        )
        # inyectar label en processed para consistencia
        processed.setdefault("label", getattr(rev, "label", None))
    return CorpusDocument(rev, processed) if processed else None