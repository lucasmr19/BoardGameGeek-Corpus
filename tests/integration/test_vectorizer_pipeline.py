import numpy as np
from src.bgg_corpus.features import ReviewVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

def test_full_pipeline_integration():
    tokens = [["amazing", "game"], ["terrible", "rules"], ["fun", "mechanics"]]
    langs = ["en", "en", "en"]
    opinions = [{"sentiment": 1}, {"sentiment": -1}, {"sentiment": 1}]
    y = np.array(["positive", "negative", "positive"])

    # Vectorización
    vec = ReviewVectorizer(max_features=20)
    X = vec.fit_transform(tokens, langs, opinions)
    assert X.shape[0] == 3

    # Model
    model = MultinomialNB()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Basic evaluation
    acc = accuracy_score(y, y_pred)
    assert acc >= 0.5