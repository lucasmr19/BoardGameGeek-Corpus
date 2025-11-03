import pytest
from src.bgg_corpus.models.review import Review

def test_review_initialization():
    review = Review(
        username="player1",
        rating=8,
        comment="Great game!",
        timestamp="2024-05-10",
        game_id="G123",
        category="positive"
    )
    assert review.username == "player1"
    assert review.comment == "Great game!"
    assert review.category == "positive"