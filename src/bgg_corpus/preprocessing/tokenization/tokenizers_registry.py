"""Registry of tokenizers and stemmers for text preprocessing.
Provides mappings from string identifiers to NLTK tokenizer and stemmer classes/functions."""

import re

from nltk.tokenize import (
    sent_tokenize,
    word_tokenize,
    WordPunctTokenizer,
    TreebankWordTokenizer,
    RegexpTokenizer,
    PunktSentenceTokenizer,
) # Add more as needed...

from nltk.stem import (
    PorterStemmer,
    SnowballStemmer,
    LancasterStemmer,
    ISRIStemmer,
    RSLPStemmer,
) # Add more as needed...

SENT_TOKENIZERS = {
    "punkt": lambda lang: lambda text: sent_tokenize(text, language=lang),
    "regex": lambda lang: lambda text: [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()],
}

WORD_TOKENIZERS = {
    "word_tokenize": lambda lang: lambda text: word_tokenize(text, language=lang),
    "treebank": lambda lang: TreebankWordTokenizer().tokenize,
    "wordpunct": lambda lang: WordPunctTokenizer().tokenize,
    "regexp": lambda lang: RegexpTokenizer(r"\w+").tokenize,
    "punkt": lambda lang: PunktSentenceTokenizer().tokenize,
}

STEMMERS = {
    "porter": PorterStemmer,
    "lancaster": LancasterStemmer,
    "snowball": lambda lang: SnowballStemmer(lang),
    "isri": ISRIStemmer,
    "rslp": RSLPStemmer,
}