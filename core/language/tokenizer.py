"""
core/language/tokenizer.py - Central place for jieba initialization
"""

# Create a new global initialization management for jieba
# We'll use this file as the central place for jieba management

from core.engine.logging import warning, debug

# Global flag and instance to track Jieba initialization
_JIEBA_INITIALIZED = False
_JIEBA_INSTANCE = None

# Import jieba for Chinese word segmentation if available
try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    warning("jieba not available, falling back to character-based tokenization for Chinese")

def initialize_jieba():
    """
    Initialize jieba if it hasn't been initialized already.
    This function should be called whenever jieba is needed.
    """
    global _JIEBA_INITIALIZED, _JIEBA_INSTANCE

    if not JIEBA_AVAILABLE:
        return None

    if not _JIEBA_INITIALIZED:
        debug(None, "Initializing Jieba dictionary (first time)")
        # Redirect stdout to suppress jieba's initialization messages
        import sys
        import io
        original_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Redirect to string buffer

        try:
            # Force loading the dictionary
            jieba.lcut("初始化")
            _JIEBA_INSTANCE = jieba
            _JIEBA_INITIALIZED = True
        finally:
            # Restore stdout
            sys.stdout = original_stdout

    return _JIEBA_INSTANCE

def get_jieba_instance():
    """
    Get the initialized jieba instance.
    Always use this function instead of importing jieba directly.
    """
    if not JIEBA_AVAILABLE:
        return None

    return initialize_jieba()

class Tokenizer:
    """Language-specific tokenization"""

    def __init__(self, config):
        """Initialize the tokenizer"""
        self.config = config
        debug(config, "Tokenizer initialized")

        # Initialize Jieba once during tokenizer initialization
        if JIEBA_AVAILABLE:
            initialize_jieba()

    def tokenize(self, text, language=None):
        """
        Tokenize text based on language

        Args:
            text (str): The text to tokenize
            language (str, optional): Language code. If None, language will be auto-detected.

        Returns:
            list: Tokens
        """
        debug(self.config, f"Tokenizing text with language: {language}")

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
        debug(self.config, "Tokenizing English text")

        # Simple whitespace tokenization for English
        import re

        # Remove punctuation and tokenize
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        tokens = text.split()

        return tokens

    def _tokenize_chinese(self, text):
        """Tokenize Chinese text"""
        debug(self.config, "Tokenizing Chinese text")

        # Use jieba for word segmentation if available
        if JIEBA_AVAILABLE:
            debug(self.config, "Using jieba for Chinese word segmentation")

            # Get the initialized jieba instance
            jieba_instance = get_jieba_instance()

            # Segment text into words
            tokens = list(jieba_instance.cut(text))

            # Filter out empty tokens and spaces
            tokens = [token for token in tokens if token.strip()]

            return tokens
        else:
            debug(self.config, "Falling back to character-based tokenization for Chinese")

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

        # Initialize Jieba once
        if self.use_jieba:
            initialize_jieba()

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
            jieba_instance = get_jieba_instance()
            tokens = list(jieba_instance.cut(text))
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