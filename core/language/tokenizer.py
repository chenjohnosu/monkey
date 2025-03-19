"""
Language-specific tokenization with enhanced Chinese support
"""

from core.engine.utils import debug_print

# Import jieba for Chinese word segmentation if available
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("jieba not available, falling back to character-based tokenization for Chinese")

class Tokenizer:
    """Language-specific tokenization"""

    def __init__(self, config):
        """Initialize the tokenizer"""
        self.config = config
        debug_print(config, "Tokenizer initialized")

    def tokenize(self, text, language=None):
        """
        Tokenize text based on language

        Args:
            text (str): The text to tokenize
            language (str, optional): Language code. If None, language will be auto-detected.

        Returns:
            list: Tokens
        """
        debug_print(self.config, f"Tokenizing text with language: {language}")

        # Skip empty text
        if not text or len(text.strip()) == 0:
            return []

        # Auto-detect language if not provided
        if language is None:
            from core.language.detector import LanguageDetector
            detector = LanguageDetector(self.config)
            language = detector.detect(text)

        # Apply language-specific tokenization
        if language == 'zh':
            return self._tokenize_chinese(text)
        else:
            return self._tokenize_english(text)

    def _tokenize_english(self, text):
        """Tokenize English text"""
        debug_print(self.config, "Tokenizing English text")

        # Simple whitespace tokenization for English
        import re

        # Remove punctuation and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()

        return tokens

    def _tokenize_chinese(self, text):
        """Tokenize Chinese text"""
        debug_print(self.config, "Tokenizing Chinese text")

        # Use jieba for word segmentation if available
        if JIEBA_AVAILABLE:
            debug_print(self.config, "Using jieba for Chinese word segmentation")
            # Segment text into words
            tokens = list(jieba.cut(text))

            # Filter out empty tokens and spaces
            tokens = [token for token in tokens if token.strip()]

            return tokens
        else:
            debug_print(self.config, "Falling back to character-based tokenization for Chinese")

            # Character-based tokenization as fallback
            tokens = []

            # Extract Chinese characters
            for char in text:
                if '\u4e00' <= char <= '\u9fff':  # Is it a Chinese character?
                    tokens.append(char)

            return tokens

    def get_ngrams(self, tokens, n=2):
        """
        Generate n-grams from tokens

        Args:
            tokens (list): Token list
            n (int): Size of n-gram

        Returns:
            list: N-grams
        """
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]