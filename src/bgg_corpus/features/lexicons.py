"""
Lexicon Loaders Module
"""

import os
from ..config import LEXICONS_DIR
from ..resources import LOGGER


# =====================================================
# ============  LEXICON LOADERS & UTILITIES  ==========
# =====================================================

class SentimentLexicon:
    """Load and manage sentiment lexicons for board game reviews."""

    def __init__(self, base_path: str = LEXICONS_DIR):
        self.base_path = base_path

        # === Base lexicons ===
        self.positive_words = self._load_lexicon("positive-words.txt", set())
        self.negative_words = self._load_lexicon("negative-words.txt", set())
        self.intensifiers = self._load_lexicon("boosters.txt", set())
        self.mitigators = self._load_lexicon("mitigators.txt", set())
        self.negation_words = self._load_lexicon("negations.txt", set())
        self.domain_terms = self._load_domain_lexicon("domain_terms.txt", set())

        # === Hedges (subfolder) ===
        self.hedge_words = self._load_lexicon("hedges/hedge_words.txt", set())
        self.relational_hedges = self._load_lexicon("hedges/relational_hedges.txt", set())
        self.discourse_markers = self._load_lexicon("hedges/discourse_markers.txt", set())
        self.propositional_hedges = self._load_lexicon("hedges/propositional_hedges.txt", set())

        # === Combined hedge set ===
        self.all_hedges = (
            self.hedge_words
            | self.relational_hedges
            | self.discourse_markers
            | self.propositional_hedges
        )

    # -----------------------------------------------------
    # ---------------- Helper Loaders ---------------------
    # -----------------------------------------------------
    def _load_lexicon(self, filename: str, fallback):
        """Load a flat lexicon (one item per line)."""
        # Ensure proper path join — supports relative or absolute paths
        if not os.path.isabs(filename):
            filepath = os.path.join(self.base_path, filename)
        else:
            filepath = filename

        # Normalize slashes for cross-platform consistency
        filepath = os.path.normpath(filepath)

        if not os.path.exists(filepath):
            LOGGER.warning(f"Lexicon file not found: {filepath}")
            return fallback

        with open(filepath, "r", encoding="utf-8") as f:
            return {
                line.strip().lower()
                for line in f
                if line.strip() and not line.startswith(";") and not line.startswith("[")
            }

    def _load_domain_lexicon(self, filename, fallback):
        """
        Load a hierarchical lexicon with [sections] from file, returning a dict.
        Format:
            [section]
            term1
            term2
        """
        filepath = os.path.join(self.base_path, filename)
        filepath = os.path.normpath(filepath)

        if not os.path.exists(filepath):
            LOGGER.warning(f"Domain lexicon file not found: {filepath}")
            return fallback

        lexicon = {}
        current_section = None

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith(";"):  # comments
                    continue
                # Detect section headers [section]
                if line.startswith("[") and line.endswith("]"):
                    current_section = line[1:-1].strip().lower()
                    lexicon[current_section] = []
                elif current_section:
                    lexicon[current_section].append(line.lower())
                else:
                    # If no section header yet, skip or store under "misc"
                    lexicon.setdefault("misc", []).append(line.lower())

        return lexicon
