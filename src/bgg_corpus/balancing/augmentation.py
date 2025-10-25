from typing import List
import nlpaug.augmenter.word as naw

from ..resources import LOGGER

class AugmentationManager:
    def __init__(self, supported_langs=None):
        if supported_langs is None:
            supported_langs = {'en', 'es', 'fr', 'de', 'it', 'pt', 'nl'}
        self.supported_langs = supported_langs
        self.augmenters = {}
        self._init_augmenters()

    def _init_augmenters(self):
        for lang in self.supported_langs:
            try:
                if lang == 'en':
                    self.augmenters[lang] = naw.SynonymAug(aug_src='wordnet')
                else:
                    # BackTranslationAug for other languages
                    self.augmenters[lang] = naw.BackTranslationAug(
                        from_model_name=f"Helsinki-NLP/opus-mt-{lang}-en",
                        to_model_name=f"Helsinki-NLP/opus-mt-en-{lang}",
                    )
                LOGGER.info(f"✓ Augmenter initialized for {lang}")
            except Exception as e:
                LOGGER.warning(f"⚠️ Could not initialize augmenter for {lang}: {e}")

    def augment(self, text: str, lang: str = 'en', num_augmentations: int = 1) -> List[str]:
        """
        Return *only* augmented variants (without including original).
        If nothing new is produced, returns [].
        """
        if not text or len(text.strip()) < 5:
            return []

        augmenter = self.augmenters.get(lang) or self.augmenters.get('en')
        if augmenter is None:
            LOGGER.debug(f"No augmenter available for {lang}")
            return []

        augmented_texts = []
        for _ in range(max(1, num_augmentations)):
            try:
                aug = augmenter.augment(text)
                # aug can be list or str
                if isinstance(aug, list):
                    for a in aug:
                        if isinstance(a, str):
                            augmented_texts.append(a)
                elif isinstance(aug, str):
                    augmented_texts.append(aug)
            except Exception as e:
                LOGGER.debug(f"Augmentation failed for lang={lang}: {e}")

        # Normalize, remove empties and strings identical to original
        cleaned = []
        for a in augmented_texts:
            if not a or not isinstance(a, str):
                continue
            a_stripped = a.strip()
            if not a_stripped or a_stripped == text.strip():
                continue
            cleaned.append(a_stripped)

        # deduplicate preserving order
        cleaned = list(dict.fromkeys(cleaned))
        if not cleaned:
            LOGGER.debug(f"Augmentation returned no new variants for text (lang={lang})")
        else:
            LOGGER.debug(f"Augmentation produced {len(cleaned)} variants for lang={lang}")

        return cleaned  # <-- do NOT include the original here

