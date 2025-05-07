"""
core/language/spacy_tokenizer.py - Unified tokenization system using spaCy
Replaces existing tokenizers with a consolidated approach
"""

import os
import warnings
from typing import List, Set, Dict, Any, Optional, Union
from pathlib import Path

from core.engine.logging import debug, warning, info, error, trace
from core.engine.utils import ensure_dir

# Global flag and instance to track spaCy initialization
_SPACY_INITIALIZED = False
_SPACY_MODELS = {}

# Check if spaCy is available
try:
    import spacy
    from spacy.language import Language

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    warning("spaCy not available, falling back to simple tokenization")

# Constants for model names
DEFAULT_EN_MODEL = "en_core_web_sm"
DEFAULT_ZH_MODEL = "zh_core_web_sm"


def initialize_spacy():
    """
    Initialize spaCy if it hasn't been initialized already.
    This function should be called whenever spaCy is needed.
    """
    global _SPACY_INITIALIZED, _SPACY_MODELS

    if not SPACY_AVAILABLE:
        warning("spaCy is not available - cannot initialize")
        return None

    if not _SPACY_INITIALIZED:
        debug(None, "Initializing spaCy models (first time)")

        # Disable warnings during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            # Initialize English model
            try:
                debug(None, f"Loading English model: {DEFAULT_EN_MODEL}")
                _SPACY_MODELS['en'] = load_spacy_model(DEFAULT_EN_MODEL)
            except Exception as e:
                error(f"Error loading English spaCy model: {str(e)}")

            # Initialize Chinese model
            try:
                debug(None, f"Loading Chinese model: {DEFAULT_ZH_MODEL}")
                _SPACY_MODELS['zh'] = load_spacy_model(DEFAULT_ZH_MODEL)
            except Exception as e:
                error(f"Error loading Chinese spaCy model: {str(e)}")

        _SPACY_INITIALIZED = True
        info(f"spaCy initialized with models: {', '.join(_SPACY_MODELS.keys())}")

    return _SPACY_MODELS

def load_spacy_model(model_name):
    """
    Load a spaCy model with automatic download if needed

    Args:
        model_name: Name of the spaCy model to load

    Returns:
        The loaded spaCy model
    """
    try:
        # Try loading the model
        return spacy.load(model_name)
    except OSError:
        # Model not found, try to download it
        info(f"spaCy model {model_name} not found, downloading...")
        try:
            spacy.cli.download(model_name)
            return spacy.load(model_name)
        except Exception as e:
            error(f"Failed to download spaCy model {model_name}: {str(e)}")

            # For Chinese, try alternative model if primary fails
            if model_name == DEFAULT_ZH_MODEL:
                try:
                    warning(f"Trying alternative Chinese model: zh_core_web_md")
                    spacy.cli.download("zh_core_web_md")
                    return spacy.load("zh_core_web_md")
                except:
                    pass

            # Create blank model as last resort
            warning(f"Creating blank model for language: {model_name[:2]}")
            return spacy.blank(model_name[:2])

def get_spacy_model(language='en'):
    """
    Get the initialized spaCy model for the specified language.

    Args:
        language: Language code ('en' or 'zh')

    Returns:
        spaCy model or None if not available
    """
    if not SPACY_AVAILABLE:
        return None

    # Initialize models if needed
    if not _SPACY_INITIALIZED:
        initialize_spacy()

    # Get model for the requested language
    if language in _SPACY_MODELS:
        return _SPACY_MODELS[language]

    # Fall back to English if the requested language is not available
    if 'en' in _SPACY_MODELS:
        warning(f"No spaCy model for language {language}, falling back to English")
        return _SPACY_MODELS['en']

    return None


def load_stopwords(language='en'):
    """
    Load stopwords for the specified language

    Args:
        language: Language code

    Returns:
        Set of stopwords
    """
    stopwords = set()

    # First try to get stopwords from spaCy
    if SPACY_AVAILABLE:
        model = get_spacy_model(language)
        if model and model.Defaults.stop_words:
            stopwords.update(model.Defaults.stop_words)

    # Then try to load from file for additional stopwords
    lexicon_dir = Path('lexicon')
    ensure_dir(lexicon_dir)

    stopwords_file = lexicon_dir / f"stopwords_{language}.txt"
    if stopwords_file.exists():
        try:
            with open(stopwords_file, 'r', encoding='utf-8') as f:
                file_stopwords = set(line.strip() for line in f if line.strip())
                stopwords.update(file_stopwords)
                debug(None, f"Loaded {len(file_stopwords)} additional stopwords for {language}")
        except Exception as e:
            warning(f"Error loading stopwords file: {str(e)}")

    return stopwords


class SpacyTokenizer:
    """Unified tokenizer for all languages using spaCy"""

    def __init__(self, config=None, stopwords=None):
        """
        Initialize the tokenizer

        Args:
            config: Configuration object
            stopwords: Optional set of stopwords to filter
        """
        self.config = config
        self.stopwords = stopwords or set()

        # Initialize spaCy
        if SPACY_AVAILABLE:
            self.models = initialize_spacy()
        else:
            self.models = {}

        debug(config, "spaCy tokenizer initialized")

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
            language = self._detect_language(text)

        # Use spaCy for tokenization if available
        if SPACY_AVAILABLE and language in self.models:
            return self._tokenize_with_spacy(text, language)

        # Fall back to basic tokenization
        if language == "zh":
            return self._basic_tokenize_chinese(text)
        else:
            return self._basic_tokenize_english(text)

    def _tokenize_with_spacy(self, text, language):
        """
        Tokenize text using spaCy

        Args:
            text (str): Text to tokenize
            language (str): Language code

        Returns:
            list: Tokens
        """
        model = self.models[language]

        # Process with spaCy
        doc = model(text)

        # Extract tokens, filtering stopwords if needed
        if self.stopwords:
            tokens = [token.text for token in doc if token.text.strip() and token.text not in self.stopwords]
        else:
            tokens = [token.text for token in doc if token.text.strip()]

        return tokens

    def _basic_tokenize_english(self, text):
        """
        Basic fallback tokenization for English text

        Args:
            text (str): Text to tokenize

        Returns:
            list: Tokens
        """
        import re

        # Simple regex-based tokenization
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        # Filter stopwords if available
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def _basic_tokenize_chinese(self, text):
        """
        Basic fallback tokenization for Chinese text

        Args:
            text (str): Text to tokenize

        Returns:
            list: Tokens
        """
        # Character-based tokenization as fallback
        tokens = []

        # Extract Chinese characters
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Is it a Chinese character?
                tokens.append(char)

        # Filter stopwords if available
        if self.stopwords:
            tokens = [token for token in tokens if token not in self.stopwords]

        return tokens

    def _detect_language(self, text):
        """
        Detect language using character distribution

        Args:
            text (str): Text to analyze

        Returns:
            str: Language code (e.g., 'en', 'zh')
        """
        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

        # Determine language based on ratio of Chinese characters
        if chinese_chars > len(text) * 0.05:
            return 'zh'
        return 'en'

    def get_ngrams(self, tokens, n=2):
        """
        Generate n-grams from tokens

        Args:
            tokens (list): Token list
            n (int): Size of n-gram

        Returns:
            list: N-grams
        """
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


class SpacyTextProcessor:
    """Processes text for analysis using spaCy"""

    def __init__(self, config):
        """Initialize the text processor with spaCy"""
        self.config = config

        # Check if spaCy should be used
        self.use_spacy = SPACY_AVAILABLE and config.get('system.use_spacy', True)

        # Initialize spaCy models
        if self.use_spacy:
            self.models = initialize_spacy()
        else:
            self.models = {}

        # Initialize stopwords
        self.english_stopwords = load_stopwords('en')
        self.chinese_stopwords = load_stopwords('zh')

        # Create tokenizer instance
        self.tokenizer = SpacyTokenizer(config)

        debug(config, "spaCy text processor initialized")

    def preprocess(self, text):
        """
        Preprocess text for analysis

        Args:
            text (str): The text to preprocess

        Returns:
            dict: Processed text information
        """
        debug(self.config, "Preprocessing text")

        # Skip empty text
        if not text or len(text.strip()) == 0:
            return {
                'original': text,
                'processed': '',
                'language': 'en',
                'length': 0,
                'tokens': 0
            }

        # Detect language
        language = self._detect_language(text)

        # Choose processing method based on available tools
        if self.use_spacy:
            debug(self.config, f"Using spaCy for {language} text processing")
            processed_text, token_count = self._preprocess_with_spacy(text, language)
        else:
            debug(self.config, f"Using traditional methods for {language} text processing")

            # Apply language-specific preprocessing
            if language == 'zh':
                processed_text = self._preprocess_chinese(text)
            else:
                processed_text = self._preprocess_english(text)

            # Count tokens
            tokens = self.tokenizer.tokenize(processed_text, language)
            token_count = len(tokens)

        return {
            'original': text,
            'processed': processed_text,
            'language': language,
            'length': len(text),
            'tokens': token_count
        }

    def _preprocess_with_spacy(self, text, language):
        """
        Preprocess text using spaCy

        Args:
            text (str): Text to preprocess
            language (str): Language code

        Returns:
            tuple: (processed_text, token_count)
        """
        model = get_spacy_model(language)
        if not model:
            # Fall back to basic preprocessing
            if language == 'zh':
                processed_text = self._preprocess_chinese(text)
            else:
                processed_text = self._preprocess_english(text)

            tokens = self.tokenizer.tokenize(processed_text, language)
            return processed_text, len(tokens)

        # Process with spaCy
        doc = model(text)

        # Remove stopwords and punctuation
        tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

        # Lemmatize for English if configured
        if language == 'en' and self.config.get('text.use_lemmatization', True):
            tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

        # Join tokens appropriately based on language
        if language == 'zh':
            processed_text = ''.join(tokens)
        else:
            processed_text = ' '.join(tokens)

        return processed_text, len(tokens)

    def _preprocess_english(self, text):
        """
        Preprocess English text using basic methods

        Args:
            text (str): Text to preprocess

        Returns:
            str: Preprocessed text
        """
        import re

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Basic tokenization and stopword removal
        words = text.split()
        words = [word for word in words if word not in self.english_stopwords]

        return ' '.join(words)

    def _preprocess_chinese(self, text):
        """
        Preprocess Chinese text using basic methods

        Args:
            text (str): Text to preprocess

        Returns:
            str: Preprocessed text
        """
        import re

        # Remove non-Chinese characters, keeping Chinese punctuation
        chinese_pattern = r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+'
        text = re.sub(chinese_pattern, ' ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove stopwords character by character
        for stopword in self.chinese_stopwords:
            text = text.replace(stopword, '')

        return text

    def _detect_language(self, text):
        """
        Detect language based on character distribution

        Args:
            text (str): Text to analyze

        Returns:
            str: Language code
        """
        # Skip empty text
        if not text or len(text.strip()) == 0:
            return 'en'

        # Count Chinese characters
        chinese_chars = sum(1 for char in text if '\u4e00' <= char <= '\u9fff')

        # Calculate ratio
        total_chars = len(text)
        chinese_ratio = chinese_chars / total_chars if total_chars > 0 else 0

        debug(self.config, f"Chinese character ratio: {chinese_ratio:.2f}")

        # If more than 5% of characters are Chinese, assume it's Chinese
        if chinese_ratio > 0.05:
            return 'zh'
        else:
            return 'en'