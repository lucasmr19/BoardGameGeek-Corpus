from collections import Counter
from typing import List, Dict, Optional, Any
from .review import Review
from .corpus_document import CorpusDocument


class GameCorpus:
    """
    Class defining the GameCorpus model for a collection of processed reviews for a specific game.
    """
    def __init__(self, game_id: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None,
                 documents: Optional[List[CorpusDocument]] = None):
        self.game_id = game_id
        self.metadata = metadata or {}
        self.documents: List[CorpusDocument] = documents or []

    def add_document(self, doc: CorpusDocument):
        doc.review.game_id = self.game_id
        doc.game_id = doc.review.game_id
        self.documents.append(doc)
    
    def count_by_category(self) -> Dict[str, int]:
        counts = Counter()
        for doc in self.documents:
            cat = getattr(doc.review, "category", None)
            if cat:
                counts[cat] += 1
        return dict(counts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "game_id": self.game_id,
            "metadata": self.metadata or {},
            "reviews": [d.to_dict() for d in self.documents]
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GameCorpus":
        gid = d.get("game_id")
        try:
            gid = int(gid) if gid is not None else None
        except Exception:
            pass
        meta = d.get("metadata", {}) or {}
        docs = []
        for r in d.get("reviews", []) or []:
            rev = Review.from_dict({**r, "game_id": gid})
            processed = {}
            if "text" in r or "patterns" in r or "clean_text" in r:
                processed["clean_text"] = r.get("clean_text")
                processed["language"] = r.get("language")
                processed.update(r.get("text", {}))
                processed["patterns"] = r.get("patterns", {})
            docs.append(CorpusDocument(rev, processed if processed else None))
        return cls(game_id=gid, metadata=meta, documents=docs)