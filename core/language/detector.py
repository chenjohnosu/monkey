"""
Language detection module with spaCy integration
"""

from core.engine.logging import debug

class LanguageDetector:
    """Detects the language of text with improved accuracy"""

    def __init__(self, config):
        """Initialize the language detector"""
        self.config = config
        debug(config, "Language detector initialized")

        # Check if spaCy is available
        self.use_spacy = False
        try:
            from core.language.spacy_tokenizer import SPACY_AVAILABLE, initialize_spacy
            if SPACY_AVAILABLE and config.get('system.use_spacy', True):
                self.use_spacy = True
                # Initialize spaCy
                initialize_spacy()
                debug(config, "Using spaCy for language detection")
        except ImportError:
            debug(config, "spaCy not available for language detection")

    def detect(self, text):
        """
        Detect the language of a given text

        Args:
            text (str): The text to analyze

        Returns:
            str: Language code (e.g., 'en', 'zh')
        """
        debug(self.config, "Detecting language")

        # Skip empty text
        if not text or len(text.strip()) == 0:
            debug(self.config, "Empty text, defaulting to English")
            return 'en'

        # Try using spaCy first if available
        if self.use_spacy:
            try:
                from core.language.spacy_tokenizer import get_spacy_model

                # Count Chinese characters for quick check
                chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')
                chinese_ratio = chinese_chars / len(text) if len(text) > 0 else 0

                # If significant Chinese characters, use Chinese model
                if chinese_ratio > 0.05:
                    debug(self.config, "Detected language: Chinese (zh)")
                    return 'zh'

                # For non-Chinese text, use English model to check
                en_model = get_spacy_model('en')
                if en_model:
                    # Use a small sample of text for efficiency
                    sample = text[:500]
                    doc = en_model(sample)

                    # Count tokens that are recognized by the model
                    recognized_tokens = sum(1 for token in doc if token.has_vector and not token.is_punct)
                    total_tokens = sum(1 for token in doc if not token.is_punct)

                    # If sufficient tokens are recognized by English model, it's likely English
                    if total_tokens > 0 and recognized_tokens / total_tokens > 0.5:
                        debug(self.config, "Detected language: English (en)")
                        return 'en'
            except Exception as e:
                debug(self.config, f"Error in spaCy language detection: {str(e)}")
                # Fall back to character-based detection

        # Character-based detection as fallback
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

        # Count English letters
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)

        # Calculate ratios
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0

        debug(self.config, f"Chinese ratio: {chinese_ratio:.2f}, English ratio: {english_ratio:.2f}")

        # If more than 5% of characters are Chinese, assume it's Chinese
        if chinese_ratio > 0.05:
            debug(self.config, "Detected language: Chinese (zh)")
            return 'zh'
        elif english_ratio > 0.3:  # If more than 30% is English alphabetic chars
            debug(self.config, "Detected language: English (en)")
            return 'en'
        else:
            # Default to English for unknown
            debug(self.config, "Unidentified language, defaulting to English (en)")
            return 'en'