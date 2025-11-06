from typing import Dict, Optional, Any

class Review:
    """
    Class defining the Review model for board game geek reviews.
    Includes methods for serialization and deserialization.
    """
    def __init__(self, username: Optional[str] = "", rating: Optional[float] = None, comment: Optional[str] = "",
                 clean_text: Optional[str] = "", timestamp: Optional[int] = None, game_id: Optional[int] = None):
        self.username = username
        self.rating = rating
        self.comment = comment or ""
        self.clean_text = clean_text
        self.timestamp = timestamp
        self.game_id = game_id
        self.label = self.rating_to_label(rating)
        self.category = self.label
    
    @staticmethod
    def rating_to_label(rating):
        """Convert numeric rating to categorical label."""
        if rating is None:
            return None
        r = float(rating)
        if r >= 7:
            return "positive"
        elif r >= 5:
            return "neutral"
        else:
            return "negative"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "rating": self.rating,
            "timestamp": self.timestamp,
            "comment": self.comment,
            "clean_text": self.clean_text,
            "game_id": self.game_id,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Review":
        username = d.get("username")
        rating = d.get("rating")
        comment = d.get("raw_text") or d.get("comment")
        clean_text = d.get("clean_text")
        timestamp = d.get("timestamp")
        gid = d.get("game_id")
        try:
            gid = int(gid) if gid is not None else None
        except Exception:
            pass
        return cls(username=username, rating=rating, comment=comment, clean_text=clean_text,
                   timestamp=timestamp, game_id=gid)