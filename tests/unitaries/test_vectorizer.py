import numpy as np
from src.bgg_corpus.features import ReviewVectorizer

def test_vectorizer_output_shapes():
    tokens = [["great", "game"], ["bad", "design"]]
    langs = ["en", "en"]
    opinions = [{"polarity": 1}, {"polarity": -1}]

    vectorizer = ReviewVectorizer(max_features=10)
    X = vectorizer.fit_transform(tokens, langs, opinions)
    
    assert X.shape[0] == 2
    assert X.shape[1] > 0