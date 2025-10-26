"""
Lexicon Loaders Module
"""

import os
from ..config import LEXICONS_DIR

# Download required NLTK resources
#required_nltk = ['vader_lexicon', 'averaged_perceptron_tagger', 'universal_tagset', 'punkt']
#for resource in required_nltk:
#    try:
#        nltk.data.find(f'tokenizers/{resource}')
#    except LookupError:
#        nltk.download(resource)



# =====================================================
# ============  LEXICON LOADERS & UTILITIES  =========
# =====================================================

class SentimentLexicon:
    """Load and manage sentiment lexicons."""
    
    def __init__(self, base_path: str = LEXICONS_DIR):
        self.base_path = base_path
        self.positive_words = self._load_positive_lexicon()
        self.negative_words = self._load_negative_lexicon()
        self.intensifiers = self._load_intensifiers()
        self.mitigators = self._load_mitigators()
        self.negation_words = self._load_negations()
        self.domain_terms = self._load_domain_terms()

    def _load_lexicon(self, filename, fallback):
        """Load a lexicon from file if available, otherwise return fallback."""
        filepath = os.path.join(self.base_path, filename)
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                return {line.strip().lower() for line in f if line.strip() and not line.startswith(';')}
        return fallback
    
    @staticmethod
    def _load_positive_lexicon():
        """Load positive sentiment words."""
        words = {
            'amazing', 'awesome', 'excellent', 'fantastic', 'great', 'good', 'wonderful',
            'love', 'enjoy', 'fun', 'engaging', 'brilliant', 'outstanding', 'superb',
            'delightful', 'thrilling', 'impressive', 'remarkable', 'beautiful', 'elegant',
            'clever', 'smart', 'intuitive', 'smooth', 'balanced', 'strategic', 'deep',
            'rewarding', 'satisfying', 'addictive', 'compelling', 'captivating', 'immersive',
            'intuitive', 'tight', 'polished', 'refined', 'intricate', 'sophisticated'
        }
        return words
    
    @staticmethod
    def _load_negative_lexicon():
        """Load negative sentiment words."""
        words = {
            'bad', 'awful', 'terrible', 'horrible', 'poor', 'weak', 'boring', 'tedious',
            'hate', 'dislike', 'frustrating', 'frustrate', 'annoying', 'annoyed', 'disappointed',
            'disappointing', 'dull', 'bland', 'broken', 'buggy', 'confusing', 'complicated',
            'convoluted', 'messy', 'messy', 'clunky', 'unbalanced', 'unplayable', 'unforgiving',
            'ugly', 'cheap', 'flimsy', 'slow', 'tedious', 'repetitive', 'shallow', 'forgettable'
        }
        return words
    
    @staticmethod
    def _load_intensifiers():
        """Load intensifying adverbs."""
        return {
            'extremely', 'very', 'so', 'really', 'quite', 'incredibly', 'absolutely',
            'utterly', 'completely', 'totally', 'entirely', 'definitely', 'certainly',
            'way', 'much', 'far', 'deeply', 'highly', 'awfully', 'incredibly', 'remarkably'
        }
    
    @staticmethod
    def _load_mitigators():
        """Load mitigating/softening adverbs."""
        return {
            'somewhat', 'kind of', 'sort of', 'rather', 'quite', 'fairly', 'reasonably',
            'almost', 'barely', 'hardly', 'scarcely', 'slightly', 'pretty', 'partially',
            'arguably', 'arguably', 'relatively', 'comparatively', 'marginally'
        }
    
    @staticmethod
    def _load_negations():
        """Load negation words."""
        return {
            'not', 'no', 'never', 'neither', 'nobody', 'nothing', 'nowhere', 'none',
            'cannot', 'can\'t', 'shouldn\'t', 'wouldn\'t', 'couldn\'t', 'won\'t',
            'isn\'t', 'aren\'t', 'wasn\'t', 'weren\'t', 'hasn\'t', 'haven\'t',
            'doesn\'t', 'don\'t', 'didn\'t', 'haven\'t', 'hadn\'t'
        }
    
    @staticmethod
    def _load_domain_terms():
        """Load board game domain-specific terms."""
        return {
            'mechanics': ['dice', 'roll', 'cards', 'tiles', 'tokens', 'board', 'pieces',
                         'meeples', 'components', 'setup', 'turn', 'round', 'phase'],
            'components': ['miniatures', 'quality', 'art', 'design', 'pieces', 'tokens',
                          'cards', 'board', 'box', 'packaging', 'production'],
            'rules': ['rulebook', 'rules', 'instructions', 'manual', 'clarity', 'flow',
                     'teaching', 'explanation', 'learning'],
            'gameplay': ['gameplay', 'playtime', 'length', 'pacing', 'turns', 'balance',
                        'replayability', 'variability', 'depth', 'strategy'],
            'theme': ['theme', 'immersion', 'atmosphere', 'setting', 'flavor', 'narrative',
                     'story', 'aesthetic']
        }