"""
Language module initialization with SpaCy as primary tokenization tool
"""

# Import SpaCy implementations (primary)
from core.language.spacy_tokenizer import (
    SPACY_AVAILABLE,
    SpacyTokenizer,
    SpacyTextProcessor,
    initialize_spacy,
    get_spacy_model,
    load_stopwords
)

# Import legacy implementation for backward compatibility
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
    # SpaCy implementations (primary)
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