"""
Helper functions for balancing the corpus.
"""

def create_augmented_review(original_review, augmented_text):
    """Create an augmented copy of a review."""
    aug_review = type(original_review)()
    for attr in ['username', 'rating', 'timestamp', 'game_id', 'category', 'label', 'fileid']:
        if hasattr(original_review, attr):
            setattr(aug_review, attr, getattr(original_review, attr))
    aug_review.comment = augmented_text
    aug_review.raw_text = augmented_text
    aug_review.is_augmented = True
    aug_review.augmented_from = id(original_review)
    return aug_review