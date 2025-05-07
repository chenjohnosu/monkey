"""
Compatibility layer for tokenization using SpaCy
Provides backward compatibility while encouraging migration to SpaCy
"""

import warnings
from typing import List, Set, Optional

# Import SpaCy implementations
from core.language.spacy_tokenizer import (
    SpacyTokenizer as BaseTokenizer,
    SPACY_AVAILABLE,
    load_stopwords,
    initialize_spacy,
    get_spacy_model
)

# Global flags
SPACY_AVAILABLE = SPACY_AVAILABLE
JIEBA_AVAILABLE = False  # We're removing jieba dependencies

def _show_deprecation_warning(method_name: str):
    """
    Generate a standardized deprecation warning

    Args:
        method_name (str): Name of the deprecated method
    """
    warnings.warn(
        f"Method '{method_name}' is deprecated. "
        "Use SpaCy tokenization methods instead.",
        DeprecationWarning,
        stacklevel=2
    )

def initialize_jieba():
    """
    Compatibility function for Jieba initialization
    Returns None and shows deprecation warning
    """
    _show_deprecation_warning("initialize_jieba")
    return None

def get_jieba_instance():
    """
    Compatibility function for getting Jieba instance
    Returns None and shows deprecation warning
    """
    _show_deprecation_warning("get_jieba_instance")
    return None

class Tokenizer:
    """
    Compatibility tokenizer that wraps SpaCy tokenization
    """
    def __init__(self, config=None, stopwords: Optional[Set[str]] = None):
        """
        Initialize the tokenizer

        Args:
            config: Configuration object (optional)
            stopwords: Set of stopwords to filter (optional)
        """
        _show_deprecation_warning("Tokenizer initialization")
        self.config = config
        self.spacy_tokenizer = BaseTokenizer(config, stopwords)

    def tokenize(self, text: str, language: Optional[str] = None) -> List[str]:
        """
        Tokenize text using SpaCy

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
            tokens (List[str]): List of tokens
            n (int): Size of n-grams

        Returns:
            List[tuple]: Generated n-grams
        """
        return self.spacy_tokenizer.get_ngrams(tokens, n)

class ChineseTokenizer:
    """
    Compatibility Chinese tokenizer using SpaCy
    """
    def __init__(self, stopwords: Optional[Set[str]] = None):
        """
        Initialize the Chinese tokenizer

        Args:
            stopwords (Set[str], optional): Set of stopwords to filter
        """
        _show_deprecation_warning("ChineseTokenizer initialization")
        self.stopwords = stopwords or set()
        self.tokenizer = BaseTokenizer(stopwords=self.stopwords)

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

        # Tokenize with language explicitly set to Chinese
        tokens = self.tokenizer.tokenize(text, language='zh')

        # Additional filtering to maintain backward compatibility
        filtered_tokens = [
            token for token in tokens
            if token not in self.stopwords
            and token.strip()
        ]

        return filtered_tokens

# Provide utility functions for backward compatibility
def load_chinese_stopwords() -> Set[str]:
    """
    Load Chinese stopwords

    Returns:
        Set[str]: Set of Chinese stopwords
    """
    return load_stopwords('zh')