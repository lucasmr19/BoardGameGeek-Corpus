#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download all necessary NLTK resources for the BoardGameGeek NLP project.
Run this script once to ensure all corpora, lexicons, and tokenizers are available.
"""

import nltk

# List of all resources your project may need
nltk_resources = [
    # Tokenizers
    "punkt",
    
    # Stopwords
    "stopwords",
    
    # POS tagging
    "averaged_perceptron_tagger",
    "universal_tagset",
    
    # Lexicons
    "vader_lexicon",
    
    # WordNet for lemmatization
    "wordnet",
    "omw-1.4",
    
    # Pronunciation dictionary for syllables/readability
    "cmudict"
] # Add more resources as needed...

for resource in nltk_resources:
    try:
        if resource in ("punkt",):
            nltk.data.find(f"tokenizers/{resource}")
        else:
            nltk.data.find(f"corpora/{resource}")
        print(f"[OK] {resource} found.")
    except LookupError:
        print(f"[INFO] Downloading {resource}...")
        nltk.download(resource, quiet=False)
