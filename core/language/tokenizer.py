"""
core/language/tokenizer.py - Compatibility layer for spaCy tokenization
Re-exports functionality from spacy_tokenizer.py for backward compatibility
"""

import warnings

# Import functionality from spacy_tokenizer
from core.language.spacy_tokenizer import (
    SPACY_AVAILABLE,
    SpacyTokenizer as NewTokenizer,  # Import with different name to avoid name conflict
    initialize_spacy,
    get_spacy_model,
    load_stopwords
)

# Set up backward compatibility flags and functions
JIEBA_AVAILABLE = False  # Mark as not available
_JIEBA_INITIALIZED = False
_JIEBA_INSTANCE = None

# Emit deprecation warning only once
_shown_warning = False

def _show_deprecation_warning():
    """Show deprecation warning once"""
    global _shown_warning
    if not _shown_warning:
        warnings.warn(
            "The tokenizer module is deprecated and will be removed in a future version. "
            "Please use SpacyTokenizer from core.language.spacy_tokenizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        _shown_warning = True

# Compatibility functions
def initialize_jieba():
    """Compatibility function that does nothing"""
    _show_deprecation_warning()
    return None

def get_jieba_instance():
    """Compatibility function that returns None"""
    _show_deprecation_warning()
    return None

# Create compatibility version of the original Tokenizer class
class Tokenizer:
    """Compatibility class that wraps SpacyTokenizer"""

    def __init__(self, config=None):
        """Initialize the tokenizer"""
        _show_deprecation_warning()
        self.config = config
        self.spacy_tokenizer = NewTokenizer(config)

    def tokenize(self, text, language=None):
        """
        Tokenize text based on language

        Args:
            text (str): The text to tokenize
            language (str, optional): Language code. If None, language will be auto-detected.

        Returns:
            list: Tokens
        """
        return self.spacy_tokenizer.tokenize(text, language)

    def get_ngrams(self, tokens, n=2):
        """
        Generate n-grams from tokens

        Args:
            tokens (list): Token list
            n (int): Size of n-gram

        Returns:
            list: N-grams
        """
        return self.spacy_tokenizer.get_ngrams(tokens, n)

# Re-export the ChineseTokenizer class that now uses spaCy internally
class ChineseTokenizer:
    """Tokenizer for Chinese text using spaCy (formerly jieba)"""

    def __init__(self, stopwords=None):
        """
        Initialize the tokenizer

        Args:
            stopwords (set, optional): Set of stopwords to filter
        """
        _show_deprecation_warning()
        self.stopwords = stopwords or set()
        self.space_chars = {" ", "　", "\u00A0", "\t", "\n", "\r", "\f", "\v"}
        if self.stopwords:
            self.stopwords.update(self.space_chars)
        else:
            self.stopwords = self.space_chars.copy()

        self.tokenizer = NewTokenizer(stopwords=self.stopwords)

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

        tokens = self.tokenizer.tokenize(text, language='zh')

        # Filter out stopwords and any whitespace tokens
        # Double filtering to ensure spaces are always removed
        tokens = [token for token in tokens
                  if token not in self.stopwords
                  and token.strip()
                  and token not in self.space_chars]

        return tokens