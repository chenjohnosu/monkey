"""
core/language/processor.py - Update TextProcessor initialization
"""

import os
import re  # Ensure re is imported for the regex operations
from core.engine.logging import debug,warning
from core.language.detector import LanguageDetector

# Import the centralized jieba availability constant and helper functions
from core.language.tokenizer import JIEBA_AVAILABLE, get_jieba_instance

# Remove the jieba import attempt here
# try:
#     import jieba
#     JIEBA_AVAILABLE = True
# except ImportError:
#     JIEBA_AVAILABLE = False
#     warning("jieba not available, falling back to character-based segmentation for Chinese")


class TextProcessor:
    """Processes text for analysis with enhanced Chinese support"""

    def __init__(self, config):
        """Initialize the text processor"""
        self.config = config
        self.language_detector = LanguageDetector(config)
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

        # Load English stopwords
        try:
            with open(os.path.join(lexicon_dir, 'stopwords_en.txt'), 'r', encoding='utf-8') as file:
                stopwords['en'] = set(line.strip() for line in file if line.strip())
            debug(self.config, f"Loaded {len(stopwords['en'])} English stopwords")
        except FileNotFoundError:
            stopwords['en'] = set()
            debug(self.config, "English stopwords file not found")

        # Load Chinese stopwords
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

        # Apply language-specific preprocessing
        if language == 'zh':
            processed_text = self._preprocess_chinese(text)
        else:
            processed_text = self._preprocess_english(text)

        # Count tokens appropriately based on language
        if language == 'zh':
            if JIEBA_AVAILABLE:
                jieba_instance = get_jieba_instance()
                if jieba_instance:
                    token_count = len(list(jieba_instance.cut(processed_text)))
                else:
                    token_count = len(processed_text)  # Character count as fallback
            else:
                token_count = len(processed_text)  # Character count as fallback
        else:
            token_count = len(processed_text.split())

        return {
            'original': text,
            'processed': processed_text,
            'language': language,
            'length': len(text),
            'tokens': token_count
        }

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