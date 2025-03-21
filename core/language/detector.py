"""
Language detection module
"""

from core.engine.logging import debug_print

class LanguageDetector:
    """Detects the language of text"""

    def __init__(self, config):
        """Initialize the language detector"""
        self.config = config
        debug_print(config, "Language detector initialized")

    def detect(self, text):
        """
        Detect the language of a given text

        Args:
            text (str): The text to analyze

        Returns:
            str: Language code (e.g., 'en', 'zh')
        """
        debug_print(self.config, "Detecting language")

        # Skip empty text
        if not text or len(text.strip()) == 0:
            debug_print(self.config, "Empty text, defaulting to English")
            return 'en'

        # Simple detection based on character set
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

        # Count English letters
        english_chars = sum(1 for char in text if char.isalpha() and ord(char) < 128)

        # Calculate ratios
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0
        english_ratio = english_chars / total_chars if total_chars > 0 else 0

        debug_print(self.config, f"Chinese ratio: {chinese_ratio:.2f}, English ratio: {english_ratio:.2f}")

        # If more than 5% of characters are Chinese, assume it's Chinese
        if chinese_ratio > 0.05:
            debug_print(self.config, "Detected language: Chinese (zh)")
            return 'zh'
        elif english_ratio > 0.3:  # If more than 30% is English alphabetic chars
            debug_print(self.config, "Detected language: English (en)")
            return 'en'
        else:
            # Default to English for unknown
            debug_print(self.config, "Unidentified language, defaulting to English (en)")
            return 'en'