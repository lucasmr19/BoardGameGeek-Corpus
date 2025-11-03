In the root project try:

pytest tests/unitaries/test_preprocessing.py -v
pytest --cov=src/bgg_corpus/preprocessing --cov-report=term-missing
