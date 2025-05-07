"""
Compatibility layer for tokenization, primarily using spaCy
"""

import warnings
from typing import List, Set, Optional

# Import spaCy implementations
from core.language.spacy_tokenizer import (
    SpacyTokenizer as NewTokenizer,
    SPACY_AVAILABLE,
    load_stopwords
)

# Deprecation warning flag
_shown_warning = False

def _show_deprecation_warning():
    """Show deprecation warning once"""
    global _shown_warning
    if not _shown_warning:
        warnings.warn(
            "This tokenizer module is deprecated. Use SpacyTokenizer from core.language.spacy_tokenizer instead.",
            DeprecationWarning,
            stacklevel=2
        )
        _shown_warning = True

# Flags for backward compatibility
SPACY_AVAILABLE = SPACY_AVAILABLE
JIEBA_AVAILABLE = False  # Set to False as jieba is no longer primary

def initialize_jieba():
    """Compatibility function that does nothing"""
    _show_deprecation_warning()
    return None

def get_jieba_instance():
    """Compatibility function that returns None"""
    _show_deprecation_warning()
    return None

class Tokenizer:
    """Compatibility tokenizer class wrapping SpacyTokenizer"""

    def __init__(self, config=None, stopwords: Optional[Set[str]] = None):
        """Initialize the tokenizer"""
        _show_deprecation_warning()
        self.config = config
        self.spacy_tokenizer = NewTokenizer(config, stopwords)

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text using SpacyTokenizer

        Args:
            text (str): Text to tokenize
            language (str, optional): Language code

        Returns:
            List[str]: Tokenized text
        """
        return self.spacy_tokenizer.tokenize(text, language)

    def get_ngrams(self, tokens: List[str], n: int = 2) -> List[tuple]:
        """
        Generate n-grams from tokens

        Args:
            tokens (List[str]): Token list
            n (int): Size of n-gram

        Returns:
            List[tuple]: N-grams
        """
        return self.spacy_tokenizer.get_ngrams(tokens, n)

class ChineseTokenizer:
    """Tokenizer for Chinese text using SpacyTokenizer"""

    def __init__(self, stopwords: Optional[Set[str]] = None):
        """
        Initialize the tokenizer

        Args:
            stopwords (Set[str], optional): Set of stopwords to filter
        """
        _show_deprecation_warning()
        self.stopwords = stopwords or set()
        self.tokenizer = NewTokenizer(stopwords=self.stopwords)

    def __call__(self, text: str) -> List[str]:
        """
        Tokenize Chinese text

        Args:
            text (str): Text to tokenize

        Returns:
            List[str]: Tokenized text
        """
        if not text:
            return []

        tokens = self.tokenizer.tokenize(text, language='zh')

        # Additional filtering to maintain backward compatibility
        tokens = [token for token in tokens
                  if token not in self.stopwords
                  and token.strip()]

        return tokens