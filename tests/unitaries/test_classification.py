import numpy as np
from sklearn.naive_bayes import MultinomialNB
from ...scripts import ModelTrainer

def test_train_and_evaluate(monkeypatch, tmp_path):
    # Simula el vectorizador y sus métodos
    class DummyVectorizer:
        def __init__(self): self.tfidf = self
        def transform(self, *_): return np.random.rand(4, 10)
        def _prefix_tokens_with_language(self, *_): return ["en_word"]
    exp = ModelTrainer(vector_dir="", dataset_dir="", output_dir=tmp_path)
    exp.vectorizer = DummyVectorizer()
    exp.n_tfidf_features = 5

    X = np.random.rand(4, 10)
    y = np.array(["pos", "neg", "neg", "pos"])
    model = MultinomialNB()
    
    exp.train_and_evaluate("NB", model, "combined", X, y, X, y, X, y)
    assert len(exp.results) > 0