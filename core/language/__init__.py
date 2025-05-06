"""
Language module initialization
"""

"""
core/language/__init__.py - Update to expose spaCy tokenizer
"""

"""
Language module initialization
"""

# Import from spacy tokenizer module
from core.language.spacy_tokenizer import (
    SPACY_AVAILABLE,
    SpacyTokenizer,
    SpacyTextProcessor,
    initialize_spacy,
    get_spacy_model,
    load_stopwords
)

# Import from existing modules for backward compatibility
from core.language.tokenizer import (
    JIEBA_AVAILABLE,
    get_jieba_instance,
    initialize_jieba,
    Tokenizer,
    ChineseTokenizer
)

from core.language.detector import LanguageDetector
from core.language.processor import TextProcessor

__all__ = [
    # spaCy implementations
    'SPACY_AVAILABLE',
    'SpacyTokenizer',
    'SpacyTextProcessor',
    'initialize_spacy',
    'get_spacy_model',
    'load_stopwords',

    # Legacy implementations for backward compatibility
    'JIEBA_AVAILABLE',
    'get_jieba_instance',
    'initialize_jieba',
    'Tokenizer',
    'ChineseTokenizer',
    'LanguageDetector',
    'TextProcessor'
]
