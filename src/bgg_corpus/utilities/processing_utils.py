from ..models import Review, CorpusDocument
from ..preprocessing import process_review_item
from ..resources import LOGGER

def process_single_review(review_item):
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
            lower=True,
            remove_stopwords=True,
            lemmatize=True,
            correct_spelling=False
        )
        # inyectar label en processed para consistencia
        processed.setdefault("label", getattr(rev, "label", None))
    return CorpusDocument(rev, processed) if processed else None

def process_review_for_parallel(review_dict):
    """
    Wrapper para procesamiento paralelo que garantiza serialización correcta.
    
    Args:
        review_dict: Diccionario con los datos del review (debe ser serializable)
    
    Returns:
        Dict con estructura: {"review": dict, "processed": dict}
        o None si el procesamiento falla completamente
    """
    try:
        # Asegurarse de que review_dict es un diccionario
        if not isinstance(review_dict, dict):
            LOGGER.error(f"[process_review_for_parallel] Expected dict, got {type(review_dict)}")
            return None
        
        # Procesar el review
        result = process_single_review(review_dict, return_dict=True)
        
        # Si el review no tiene comentario o falla el procesamiento
        if result is None:
            LOGGER.warning(f"[process_review_for_parallel] Review returned None for game_id={review_dict.get('game_id')}")
            return None
        
        return result
        
    except Exception as e:
        LOGGER.error(f"[process_review_for_parallel] Error processing review: {e}")
        LOGGER.error(f"[process_review_for_parallel] Review data: {review_dict}")
        return None