"""
Language-specific tokenization with enhanced Chinese support
"""

from core.engine.logging import debug_print,warning

# Global flag to track Jieba initialization
_JIEBA_INITIALIZED = False

# Import jieba for Chinese word segmentation if available
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    warning("jieba not available, falling back to character-based tokenization for Chinese")

class Tokenizer:
    """Language-specific tokenization"""

    def __init__(self, config):
        """Initialize the tokenizer"""
        self.config = config
        debug_print(config, "Tokenizer initialized")

        # Initialize Jieba if needed
        if JIEBA_AVAILABLE:
            self._ensure_jieba_initialized()

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

            # Ensure Jieba is initialized
            self._ensure_jieba_initialized()

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

    @staticmethod
    def _ensure_jieba_initialized():
        """Ensure Jieba is initialized only once"""
        global _JIEBA_INITIALIZED

        if not _JIEBA_INITIALIZED and JIEBA_AVAILABLE:
            debug_print(None, "Initializing Jieba dictionary (first time)")
            # Jieba is lazy-loaded, accessing any function will initialize it
            jieba.lcut("初始化")  # This forces Jieba to load its dictionary
            _JIEBA_INITIALIZED = True

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


class ChineseTokenizer:
    """Tokenizer for Chinese text using jieba if available"""

    def __init__(self, stopwords=None):
        """
        Initialize the tokenizer

        Args:
            stopwords (set, optional): Set of stopwords to filter
        """
        self.use_jieba = JIEBA_AVAILABLE
        self.stopwords = stopwords or set()

        # Always treat spaces as stopwords regardless of what's passed in
        self.space_chars = {" ", "　", "\u00A0", "\t", "\n", "\r", "\f", "\v"}
        if self.stopwords:
            self.stopwords.update(self.space_chars)
        else:
            self.stopwords = self.space_chars.copy()

        # Initialize Jieba if needed
        if self.use_jieba:
            # Use the static method from Tokenizer
            Tokenizer._ensure_jieba_initialized()

    def __call__(self, text):
        """
        Tokenize Chinese text and filter stopwords

        Args:
            text (str): Text to tokenize

        Returns:
            list: List of tokens
        """
        if not text:
            return []

        # Tokenize based on available libraries
        if self.use_jieba:
            tokens = list(jieba.cut(text))
        else:
            # Character-based fallback
            tokens = [char for char in text if '\u4e00' <= char <= '\u9fff']

        # Filter out stopwords and any whitespace tokens
        # Double filtering to ensure spaces are always removed
        tokens = [token for token in tokens
                  if token not in self.stopwords
                  and token.strip()
                  and token not in self.space_chars]

        return tokens