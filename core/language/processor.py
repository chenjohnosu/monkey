"""
core/language/processor.py - Update TextProcessor to use spaCy
"""

import os
import re
from core.engine.logging import debug, warning, info, error
from core.language.detector import LanguageDetector

# Import spaCy functionality if available
try:
    from core.language.spacy_tokenizer import (
        SPACY_AVAILABLE,
        initialize_spacy,
        get_spacy_model,
        load_stopwords
    )
except ImportError:
    SPACY_AVAILABLE = False
    warning("spaCy integration not available, falling back to traditional methods")

# Keep backward compatibility with jieba
from core.language.tokenizer import JIEBA_AVAILABLE, get_jieba_instance


class TextProcessor:
    """Processes text for analysis with enhanced language support using spaCy where available"""

    def __init__(self, config):
        """Initialize the text processor"""
        self.config = config
        self.language_detector = LanguageDetector(config)

        # Initialize spaCy if available
        self.use_spacy = SPACY_AVAILABLE and config.get('system.use_spacy', True)
        if self.use_spacy:
            try:
                initialize_spacy()
                debug(config, "Using spaCy for text processing")
            except Exception as e:
                error(f"Error initializing spaCy: {str(e)}")
                self.use_spacy = False

        # Load stopwords
        self.stopwords = self._load_stopwords()

        debug(config, "Text processor initialized")

    def _load_stopwords(self):
        """Load stopwords for supported languages"""
        stopwords = {}

        # Define lexicon directory
        lexicon_dir = 'lexicon'
        if not os.path.exists(lexicon_dir):
            os.makedirs(lexicon_dir)
            debug(self.config, f"Created lexicon directory: {lexicon_dir}")

        # Try to use spaCy stopwords first
        if self.use_spacy:
            try:
                stopwords['en'] = load_stopwords('en')
                stopwords['zh'] = load_stopwords('zh')
                debug(self.config, f"Loaded stopwords from spaCy models")
                return stopwords
            except Exception as e:
                warning(f"Could not load spaCy stopwords: {str(e)}")

        # Fallback: Load from files
        try:
            with open(os.path.join(lexicon_dir, 'stopwords_en.txt'), 'r', encoding='utf-8') as file:
                stopwords['en'] = set(line.strip() for line in file if line.strip())
            debug(self.config, f"Loaded {len(stopwords['en'])} English stopwords")
        except FileNotFoundError:
            stopwords['en'] = set()
            debug(self.config, "English stopwords file not found")

        try:
            with open(os.path.join(lexicon_dir, 'stopwords_zh.txt'), 'r', encoding='utf-8') as file:
                stopwords['zh'] = set(line.strip() for line in file if line.strip())
            debug(self.config, f"Loaded {len(stopwords['zh'])} Chinese stopwords")
        except FileNotFoundError:
            stopwords['zh'] = set()
            debug(self.config, "Chinese stopwords file not found")

        return stopwords

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
        language = self.language_detector.detect(text)

        # Choose processing method based on available tools
        if self.use_spacy:
            # Use spaCy-based processing
            processed_text, token_count = self._preprocess_with_spacy(text, language)
        else:
            # Apply traditional language-specific preprocessing
            if language == 'zh':
                processed_text = self._preprocess_chinese(text)
            else:
                processed_text = self._preprocess_english(text)

            # Count tokens appropriately based on language
            token_count = self._count_tokens(processed_text, language)

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
            text (str): Text to process
            language (str): Language code

        Returns:
            Tuple[str, int]: Processed text and token count
        """
        try:
            # Get appropriate spaCy model
            nlp = get_spacy_model(language)
            if not nlp:
                # Fall back to traditional methods if model is unavailable
                if language == 'zh':
                    return self._preprocess_chinese(text), self._count_tokens(text, language)
                else:
                    return self._preprocess_english(text), self._count_tokens(text, language)

            # Process with spaCy
            doc = nlp(text)

            # Filter out stopwords and punctuation
            tokens = [token.text for token in doc if not token.is_stop and not token.is_punct]

            # Join tokens appropriately based on language
            if language == 'zh':
                processed_text = ''.join(tokens)
            else:
                processed_text = ' '.join(tokens)

            return processed_text, len(tokens)

        except Exception as e:
            debug(self.config, f"spaCy processing error: {str(e)}")
            # Fall back to traditional methods
            if language == 'zh':
                return self._preprocess_chinese(text), self._count_tokens(text, language)
            else:
                return self._preprocess_english(text), self._count_tokens(text, language)

    def _preprocess_english(self, text):
        """Preprocess English text"""
        debug(self.config, "Preprocessing English text")

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Basic tokenization and stopword removal
        words = text.split()
        words = [word for word in words if word not in self.stopwords.get('en', set())]

        return ' '.join(words)

    def _preprocess_chinese(self, text):
        """Preprocess Chinese text"""
        debug(self.config, "Preprocessing Chinese text")

        # Remove non-Chinese characters, keeping Chinese punctuation
        # This regex keeps Chinese characters, punctuation and some common symbols
        chinese_pattern = r'[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+'
        text = re.sub(chinese_pattern, ' ', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove stopwords if available
        if 'zh' in self.stopwords and self.stopwords['zh']:
            if JIEBA_AVAILABLE:
                # Use jieba for word segmentation using our centralized instance
                jieba_instance = get_jieba_instance()
                if jieba_instance:
                    segments = list(jieba_instance.cut(text))
                    # Filter out stopwords
                    segments = [word for word in segments if word not in self.stopwords['zh']]
                    text = ''.join(segments)
                else:
                    # Character-based stopword removal as fallback
                    for stopword in self.stopwords['zh']:
                        text = text.replace(stopword, '')
            else:
                # Character-based stopword removal as fallback
                for stopword in self.stopwords['zh']:
                    text = text.replace(stopword, '')

        return text

    def _count_tokens(self, text, language):
        """
        Count tokens in processed text

        Args:
            text (str): Processed text
            language (str): Language code

        Returns:
            int: Token count
        """
        if language == 'zh':
            if JIEBA_AVAILABLE:
                jieba_instance = get_jieba_instance()
                if jieba_instance:
                    return len(list(jieba_instance.cut(text)))
                else:
                    return len(text)  # Character count as fallback
            else:
                return len(text)  # Character count as fallback
        else:
            return len(text.split())