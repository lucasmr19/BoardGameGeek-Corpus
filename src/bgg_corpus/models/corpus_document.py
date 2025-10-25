from typing import Dict, Optional, Any
from .review import Review


class CorpusDocument:
    """Class defining the CorpusDocument model for processed reviews."""
    def __init__(self, review: Review, processed: Optional[Dict[str, Any]] = None):
        self.review = review
        self.raw_text = review.comment
        self.clean_text = None
        self.language = None

        # Estructura de texto / tokens
        self.processed = {
            "sentences": [], "tokens": [], "tokens_no_stopwords": [], "stems": [], "lemmas": [],
            "pos_tags": [], "dependencies": []
        }

        # Patrones extraídos
        self.patterns = {
            "emails": [], "dates": [], "phones": [], "hashtags": [], "mentions": [],
            "urls": [], "emojis": []
        }

        self.game_id = review.game_id
        self.category = review.category
        self.linguistic_features = {}

        if processed:
            self.clean_text = processed.get("clean_text")
            self.language = processed.get("language")
            
            self.processed.update({
                "sentences": processed.get("sentences", []),
                "tokens": processed.get("tokens", []),
                "tokens_no_stopwords": processed.get("tokens_no_stopwords", []),
                "stems": processed.get("stems", []),
                "lemmas": processed.get("lemmas", []),
                "pos_tags": processed.get("pos_tags", []),
                "dependencies": processed.get("dependencies", []),
            })
            
            patterns_dict = processed.get("patterns", {})
            for k in self.patterns:
                self.patterns[k] = patterns_dict.get(k, [])
            
            self.linguistic_features = processed.get("linguistic_features", {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.review.username,
            "rating": self.review.rating,
            "timestamp": self.review.timestamp,
            "game_id": self.game_id,
            "category": self.category,
            "raw_text": self.raw_text,
            "clean_text": self.clean_text,
            "language": self.language,
            "processed": self.processed,
            "linguistic_features": self.linguistic_features,
            "patterns": self.patterns,
        }

    def to_feature_dict(self) -> Dict[str, Any]:
        """
        Devuelve solo las características lingüísticas relevantes para vectorización.
        """
        feats = self.linguistic_features or {}

        # Si quisieras incluir metadatos numéricos adicionales, se podrían añadir aquí
        feats_out = {}

        for k, v in feats.items():
            # En caso de que las features estén anidadas (por ejemplo 'vader_scores', 'domain_specific')
            if isinstance(v, dict):
                for sub_k, sub_v in v.items():
                    feats_out[f"{k}_{sub_k}"] = sub_v
            else:
                feats_out[k] = v

        return feats_out
