import logging
import sys
import time
import re
import nltk
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter
import torch
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.collocations import TrigramAssocMeasures, TrigramCollocationFinder
from llama_index.core import StorageContext, load_index_from_storage

from config import MonkeyConfig

# Import for Chinese language support
try:
    import jieba
    import langdetect

    CHINESE_SUPPORT = True
except ImportError:
    CHINESE_SUPPORT = False
    print("Chinese support libraries (jieba, langdetect) not installed. Chinese processing will be limited.")


class ThematicAnalyzer:
    """Performs comprehensive thematic analysis across all documents with multilingual support."""

    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load spaCy models
        try:
            self.nlp = spacy.load('en_core_web_sm')
            if CHINESE_SUPPORT:
                try:
                    self.zh_nlp = spacy.load('zh_core_web_sm')
                except (OSError, IOError):
                    self.zh_nlp = None
                    self.logger.info("Chinese spaCy model not available. Using jieba for Chinese processing.")
        except Exception as e:
            self.logger.error(f"Error loading spaCy models: {str(e)}")
            self.nlp = None
            self.zh_nlp = None

        # Download NLTK resources
        try:
            stopwords.words('english')
        except LookupError:
            nltk.download('stopwords')
        try:
            word_tokenize('test')
        except LookupError:
            nltk.download('punkt')

        # Initialize stopwords
        self.english_stop_words = self._get_enhanced_english_stopwords()
        self.chinese_stop_words = self._get_chinese_stopwords() if CHINESE_SUPPORT else set()

        print(f"Initialized ThematicAnalyzer with config: vdb_dir={config.vdb_dir}")
        print(f"Chinese language support: {'Available' if CHINESE_SUPPORT else 'Not available'}")

    def _get_enhanced_english_stopwords(self) -> set:
        """Create an enhanced set of stop words for transcript analysis in English."""
        # Get standard stopwords from NLTK and spaCy
        stop_words = set(stopwords.words('english'))
        if self.nlp:
            spacy_stops = set(self.nlp.Defaults.stop_words)
        else:
            spacy_stops = set()

        # Custom stopwords for transcripts and interviews
        transcript_stops = {
            # Common speech fillers and discourse markers
            'um', 'uh', 'hmm', 'like', 'you know', 'i mean', 'so', 'well', 'anyway',
            'actually', 'basically', 'literally', 'obviously', 'right', 'okay',

            # Transcript notation
            'inaudible', 'crosstalk', 'pause', 'laughter', 'applause', 'silence',
            'break', 'interruption', 'overlapping', 'speaker', 'speaking',

            # Time references
            'second', 'minute', 'hour', 'day', 'week', 'month', 'year',
            'morning', 'afternoon', 'evening', 'night', 'today', 'yesterday', 'tomorrow',

            # Speaker identifiers
            'interviewer', 'interviewee', 'moderator', 'participant', 'respondent',
            'subject', 'speaker', 'speaker1', 'speaker2', 'person',

            # Common transcript words
            'said', 'says', 'saying', 'tell', 'told', 'ask', 'asked', 'asking',
            'talk', 'talked', 'talking', 'think', 'thought', 'thinking',
            'get', 'got', 'getting', 'go', 'going', 'went', 'come', 'came', 'coming',

            # Speech reporting verbs
            'mention', 'mentioned', 'state', 'stated', 'explain', 'explained',
            'describe', 'described', 'indicate', 'indicated', 'suggest', 'suggested',
            'claim', 'claimed', 'argue', 'argued', 'agree', 'agreed', 'deny', 'denied',

            # Numbers and units as words
            'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            'first', 'second', 'third', 'fourth', 'fifth', 'hundred', 'thousand', 'million'
        }

        # Combine all stop words
        all_stops = stop_words.union(spacy_stops).union(transcript_stops)
        return all_stops

    def _get_chinese_stopwords(self) -> set:
        """Load Chinese stop words from a predefined list."""
        if not CHINESE_SUPPORT:
            return set()

        # Common Chinese stop words
        chinese_stops = {
            # Basic particles and functional words
            '的', '了', '和', '是', '就', '都', '而', '及', '與', '著', '或', '一個', '沒有',
            '我們', '你們', '他們', '她們', '這個', '那個', '這些', '那些', '不', '在',
            '人', '我', '來', '他', '上', '個', '到', '說', '們', '為', '子', '你',
            '有', '那', '這', '就', '以', '等', '著', '把', '才', '地', '得', '於',

            # Pronouns
            '您', '妳', '我', '我們', '你', '你們', '他', '她', '它', '他們', '她們', '它們',
            '咱們', '誰', '哪個', '這', '那', '這些', '那些', '誰們', '啥',

            # Numbers and measure words
            '一', '二', '三', '幾', '多少', '個', '些', '只', '條', '種', '樣', '件',

            # Adverbs
            '很', '非常', '太', '更', '比較', '越來越', '稍', '略', '幾乎', '基本',
            '相當', '稍微', '極其', '極端', '最', '最為', '多麼', '大約', '不太',

            # Conjunctions
            '和', '跟', '與', '以及', '並且', '而且', '況且', '但是', '然而', '不過',
            '可是', '雖然', '盡管', '因為', '由於', '所以', '因此', '是故', '於是',

            # Prepositions
            '在', '把', '將', '對於', '關於', '向', '往', '依', '靠', '按照', '根據',

            # Time words
            '現在', '當時', '曾經', '剛才', '過去', '將來', '終於', '最終', '始終',
            '已經', '剛剛', '後來', '從來', '以前', '之前', '當初', '將近', '昨天',

            # Auxiliary words
            '的', '地', '得', '所', '似的', '般', '樣', '一般', '似', '如同', '這樣',
            '那樣', '如此', '之', '者', '所', '等', '等等',

            # Modal particles
            '吧', '呢', '啊', '嗎', '呀', '哇', '哦', '喔', '喲', '哎', '誒', '欸',
            '嘿', '嗨', '嘛', '哼', '哈', '啦', '咧', '嘮', '囉', '唷',

            # Common verbs often acting as function words
            '是', '有', '要', '會', '能', '可以', '應該', '可能', '應當', '須', '得',
            '該', '需要', '值得', '敢', '肯', '愿意', '想', '想要',

            # Academic-specific
            '研究', '論文', '分析', '調查', '實驗', '數據', '結果', '方法', '表明',
            '表示', '顯示', '認為', '指出', '發現', '證實', '對比', '比較', '總結',

            # Measurements and units
            '公分', '厘米', '千米', '公里', '米', '毫米', '微米', '釐米', '吋', '英寸',
            '英尺', '呎', '碼', '公斤', '千克', '克', '毫克', '微克', '噸', '盎司',
            '磅', '秒', '分鐘', '小時', '刻', '天', '週', '月', '年', '世紀',

            # Miscellaneous common words
            '例如', '比如', '像', '假如', '要是', '如果', '若', '如何', '怎麼', '怎樣',
            '多久', '哪里', '哪裡', '何處', '何時', '何人', '何物', '何事', '何故',
            '為何', '為什麼', '怎麼樣', '多長時間', '多遠', '多大', '多重', '多少'

        }

        # Add Chinese transcript-specific stopwords
        transcript_stops = {
            # Transcript-specific words (Chinese equivalent of um, uh, etc.)
            '啊', '嗯', '呃', '那個', '這個', '就是', '就是說', '也就是說', '嗯嗯',
            '然後', '所以', '但是', '因為', '如果', '的話', '就像', '比如說',

            # Chinese filler words
            '其實', '基本上', '可能', '應該', '大概', '大約', '差不多', '基本上',

            # Time-related
            '分鐘', '小時', '點鐘', '年', '月', '日', '星期', '禮拜', '早上', '上午',
            '下午', '晚上', '凌晨', '半夜', '今天', '昨天', '明天', '前天', '後天',

            # Custome
            '語者', '對呀', '是啊'
        }

        chinese_stops.update(transcript_stops)

        # Try to load from external file if available
        try:
            stopwords_file = Path('chinese_stopwords.txt')
            if stopwords_file.exists():
                with open(stopwords_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip()
                        if word:
                            chinese_stops.add(word)
        except Exception as e:
            self.logger.warning(f"Error loading Chinese stopwords file: {e}")

        return chinese_stops

    def detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        if not CHINESE_SUPPORT:
            return "en"  # Default to English if langdetect is not available

        if not text or len(text.strip()) < 10:
            return "en"  # Default to English for very short or empty text

        try:
            # Use a sample of the text for faster detection
            sample = text[:min(len(text), 1000)]
            # Count Chinese characters for a quick check
            chinese_char_count = sum(1 for char in sample if '\u4e00' <= char <= '\u9fff')

            # If more than 20% of characters are Chinese, assume Chinese
            if chinese_char_count > len(sample) * 0.2:
                return 'zh'

            # Otherwise use langdetect
            lang = langdetect.detect(sample)

            # Normalize Chinese language codes
            if lang in ['zh-cn', 'zh-tw', 'zh']:
                return 'zh'
            return lang
        except:
            # Default to English if detection fails
            return "en"

    def _clean_text_for_analysis(self, text: str) -> str:
        """
        Clean text specifically for thematic analysis, removing timestamps,
        numbers, and other patterns that might skew analysis.
        """
        if not text:
            return ""

        # Detect language
        language = self.detect_language(text)

        # Remove timestamp patterns (various formats)
        text = re.sub(r'\d{1,2}:\d{2}(:\d{2})?', '', text)  # HH:MM:SS or HH:MM
        text = re.sub(r'\d{1,2}[:.]\d{2}[:.]\d{2}', '', text)  # HH.MM.SS

        # Remove date patterns
        text = re.sub(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', '', text)  # MM/DD/YYYY, DD/MM/YYYY
        text = re.sub(r'\d{4}[/-]\d{1,2}[/-]\d{1,2}', '', text)  # YYYY/MM/DD

        # Remove common transcript markers
        text = re.sub(r'\[[^\]]*\]', '', text)  # [inaudible], [laughter], etc.
        text = re.sub(r'\([^)]*\)', '', text)  # (pause), (crosstalk), etc.

        # Replace speaker identifiers
        text = re.sub(r'Speaker\s*\d+:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Person\s*[A-Z]:', '', text, flags=re.IGNORECASE)

        # Language-specific cleaning
        if language == 'zh':
            # For Chinese, remove specific punctuation and symbols
            text = re.sub(r'[「」『』【】《》〈〉""''（）]', '', text)

        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text)

        return text.strip()

    def _preprocess_text(self, text: str, language: str = None) -> List[str]:
        """
        Preprocess text for analysis, extracting key terms with language detection.

        Args:
            text (str): The text to preprocess
            language (str): Explicit language code ('en', 'zh') or None for auto-detection

        Returns:
            List[str]: Processed tokens
        """
        # Clean text first
        text = self._clean_text_for_analysis(text)

        if not text:
            return []

        # Detect language if not specified
        if language is None:
            language = self.detect_language(text)

        # Process based on language
        if language == 'zh':
            return self._preprocess_chinese_text(text)
        else:
            return self._preprocess_english_text(text)

    def _preprocess_english_text(self, text: str) -> List[str]:
        """Preprocess English text with stop word removal and cleaning."""
        # Remove standalone numbers
        text = re.sub(r'\b\d+\b', '', text)

        # Remove any remaining digits mixed with letters
        text = re.sub(r'\b\w*\d+\w*\b', '', text)

        # Tokenize text
        tokens = word_tokenize(text.lower())

        # Remove stopwords and punctuation
        tokens = [token for token in tokens
                  if token.isalnum()
                  and token not in self.english_stop_words
                  and len(token) > 2]  # Filter out very short tokens

        # Use spaCy for advanced processing if available
        if self.nlp:
            doc = self.nlp(" ".join(tokens))
            lemmas = [
                token.lemma_ for token in doc
                if not token.is_stop
                   and not token.is_punct
                   and not token.like_num
            ]
            return lemmas
        else:
            return tokens

    def _preprocess_chinese_text(self, text: str) -> List[str]:
        """
        Specialized preprocessing for Chinese text with jieba segmentation.
        """
        if not CHINESE_SUPPORT:
            self.logger.warning("Chinese support not available. Returning empty list.")
            return []

        try:
            # Remove any remaining digits (including Chinese numerals)
            text = re.sub(r'\d+', '', text)
            text = re.sub(r'[零一二三四五六七八九十百千万亿]+', '', text)

            # Use jieba for Chinese word segmentation
            words = jieba.lcut(text)

            # Remove stop words and empty strings
            filtered_words = [
                word for word in words
                if word not in self.chinese_stop_words
                   and not re.match(r'^[\s\d]+$', word)  # Remove spaces and numbers
                   and len(word.strip()) > 0
            ]

            return filtered_words

        except Exception as e:
            self.logger.error(f"Error in Chinese text preprocessing: {str(e)}")
            return []

    def _extract_key_phrases(self, text: str, top_n: int = 20, language: str = None) -> List[str]:
        """
        Extract meaningful key phrases from text with language detection.

        Args:
            text (str): The text to analyze
            top_n (int): Number of top phrases to return
            language (str): Explicit language code or None for auto-detection

        Returns:
            List[str]: Top key phrases
        """
        # Clean the text
        text = self._clean_text_for_analysis(text)

        if not text or len(text) < 10:
            return []

        # Detect language if not specified
        if language is None:
            language = self.detect_language(text)

        # Extract phrases based on language
        if language == 'zh':
            return self._extract_chinese_key_phrases(text, top_n)
        else:
            return self._extract_english_key_phrases(text, top_n)

    def _extract_english_key_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """Extract key phrases from English text using NLTK and spaCy."""
        if not self.nlp:
            return []

        doc = self.nlp(text)

        # Get noun chunks (noun phrases) with filtering
        chunks = []
        for chunk in doc.noun_chunks:
            # Skip chunks that are just numbers or very short
            if not re.match(r'^\d+$', chunk.text) and len(chunk.text.strip()) > 3:
                # Check if the chunk contains mostly stopwords
                words = [token.text.lower() for token in chunk if token.is_alpha]
                stopwords_cnt = sum(1 for w in words if w in self.english_stop_words)

                # Only add if less than half are stopwords
                if words and stopwords_cnt / len(words) < 0.5:
                    chunks.append(chunk.text.lower())

        # Get named entities with filtering
        entities = []
        for ent in doc.ents:
            # Skip date and time entities, and numeric entities
            if ent.label_ not in ['DATE', 'TIME', 'CARDINAL', 'ORDINAL', 'QUANTITY', 'PERCENT']:
                entities.append(ent.text.lower())

        # Use NLTK's collocation finders for additional phrases
        words = self._preprocess_english_text(text)

        # Extract bigrams
        bigram_measures = BigramAssocMeasures()
        bigram_finder = BigramCollocationFinder.from_words(words)
        # Filter out bigrams with stopwords
        bigram_finder.apply_freq_filter(3)  # Minimum frequency
        bigrams = bigram_finder.nbest(bigram_measures.pmi, 20)
        bigram_phrases = [' '.join(bigram) for bigram in bigrams]

        # Extract trigrams
        trigram_measures = TrigramAssocMeasures()
        trigram_finder = TrigramCollocationFinder.from_words(words)
        # Filter out trigrams with stopwords
        trigram_finder.apply_freq_filter(2)  # Minimum frequency
        trigrams = trigram_finder.nbest(trigram_measures.pmi, 15)
        trigram_phrases = [' '.join(trigram) for trigram in trigrams]

        # Combine and count occurrences
        all_phrases = chunks + entities + bigram_phrases + trigram_phrases
        phrase_counts = Counter(all_phrases)

        # Filter out phrases that are just numbers or very short
        filtered_phrases = {phrase: count for phrase, count in phrase_counts.items()
                            if not re.match(r'^\d+$', phrase) and len(phrase.strip()) > 3}

        # Return most common phrases
        return [phrase for phrase, _ in Counter(filtered_phrases).most_common(top_n)]

    def _extract_chinese_key_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """Extract key phrases from Chinese text."""
        if not CHINESE_SUPPORT:
            return []

        try:
            # Use jieba to extract key phrases
            import jieba.analyse

            # Extract using TextRank algorithm (more context-aware than TF-IDF)
            keywords = jieba.analyse.textrank(
                text,
                topK=top_n * 2,  # Extract more and filter
                allowPOS=('ns', 'n', 'vn', 'v')  # Allow nouns, verb-nouns, and verbs
            )

            # Filter out single-character words and numbers
            filtered_keywords = [
                kw for kw in keywords
                if len(kw) > 1 and not re.match(r'^\d+$', kw)
            ]

            # Try to extract noun phrases if spaCy Chinese model is available
            noun_phrases = []
            if self.zh_nlp:
                doc = self.zh_nlp(text)
                for chunk in doc.noun_chunks:
                    if len(chunk.text) > 1:
                        noun_phrases.append(chunk.text)

            # Combine keywords and noun phrases, prioritizing noun phrases
            all_phrases = noun_phrases + filtered_keywords

            # Remove duplicates while preserving order
            seen = set()
            unique_phrases = [
                phrase for phrase in all_phrases
                if not (phrase in seen or seen.add(phrase))
            ]

            return unique_phrases[:top_n]

        except Exception as e:
            self.logger.error(f"Error extracting Chinese key phrases: {str(e)}")
            return []

    def _load_documents_from_vector_store(self) -> List[Dict[str, Any]]:
        """Load documents from existing vector store with metadata."""
        try:
            print(f"Loading documents from vector store: {self.config.vdb_dir}")
            storage_context = StorageContext.from_defaults(
                persist_dir=self.config.vdb_dir
            )
            index = load_index_from_storage(storage_context)

            # Extract text and metadata from all nodes in the index
            documents = []
            for node_id, node in index.storage_context.docstore.docs.items():
                if hasattr(node, 'text'):
                    doc_info = {
                        'id': node_id,
                        'text': node.text,
                        'metadata': {},
                        'language': None  # Will be detected during processing
                    }

                    # Extract metadata if available
                    if hasattr(node, 'metadata'):
                        doc_info['metadata'] = node.metadata

                    documents.append(doc_info)

            print(f"Loaded {len(documents)} document chunks from vector store")

            # Detect languages for all documents
            if CHINESE_SUPPORT:
                print("Detecting document languages...")
                language_counts = {"en": 0, "zh": 0, "other": 0}

                for doc in documents:
                    # Detect language (using just a sample for efficiency)
                    sample = doc['text'][:min(500, len(doc['text']))]
                    lang = self.detect_language(sample)
                    doc['language'] = lang

                    # Count languages
                    if lang == 'en':
                        language_counts['en'] += 1
                    elif lang == 'zh':
                        language_counts['zh'] += 1
                    else:
                        language_counts['other'] += 1

                print(f"Document language distribution: English: {language_counts['en']}, "
                      f"Chinese: {language_counts['zh']}, Other: {language_counts['other']}")

            return documents

        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            print(f"Error loading vector store: {str(e)}")
            return []

    def _preprocess_documents(self) -> List[Dict[str, Any]]:
        """Preprocess all documents for thematic analysis with enhanced cleaning."""
        documents = self._load_documents_from_vector_store()
        processed_docs = []

        print("Preprocessing documents for thematic analysis...")
        for doc in documents:
            # Get document language (if already detected)
            language = doc.get('language')

            # Apply thorough cleaning before analysis
            cleaned_text = self._clean_text_for_analysis(doc['text'])

            if cleaned_text:
                processed_doc = doc.copy()
                processed_doc['text'] = cleaned_text
                # Detect language if not already done
                if not language:
                    processed_doc['language'] = self.detect_language(cleaned_text)
                processed_docs.append(processed_doc)

        print(f"Preprocessed {len(processed_docs)} documents out of {len(documents)}")
        return processed_docs

    def extract_common_keyphrases(self, min_doc_frequency: int = 3) -> Dict[str, Any]:
        """
        Extract common key phrases across all documents and categorize them.
        Uses NLTK and spaCy for extracting phrases instead of Gensim.

        Args:
            min_doc_frequency: Minimum number of documents where a term should appear

        Returns:
            Dict with language-specific key phrases and metrics
        """
        documents = self._preprocess_documents()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Group documents by language
        en_docs = []
        zh_docs = []
        other_docs = []

        for doc in documents:
            lang = doc.get('language', 'en')
            if lang == 'en':
                en_docs.append(doc['text'])
            elif lang == 'zh':
                zh_docs.append(doc['text'])
            else:
                other_docs.append(doc['text'])

        print(f"Document language distribution: English: {len(en_docs)}, "
              f"Chinese: {len(zh_docs)}, Other: {len(other_docs)}")

        # Process English documents
        en_keyphrases = {}
        if en_docs:
            print(f"Extracting key phrases from {len(en_docs)} English documents...")

            # Create document-term matrix with TF-IDF
            en_vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=min_doc_frequency,
                max_df=0.85,
                stop_words='english',
                ngram_range=(1, 3)  # Include unigrams, bigrams, and trigrams
            )

            try:
                en_tfidf = en_vectorizer.fit_transform(en_docs)

                # Get high TF-IDF terms
                feature_names = en_vectorizer.get_feature_names_out()

                # Calculate average TF-IDF scores across documents
                avg_tfidf = np.asarray(en_tfidf.mean(axis=0)).flatten()

                # Get top terms by average TF-IDF
                tfidf_scores = [(feature_names[i], avg_tfidf[i])
                                for i in avg_tfidf.argsort()[::-1][:100]]

                # Extract phrases using spaCy-based method
                custom_phrases = []
                for doc_text in en_docs[:min(20, len(en_docs))]:  # Process subset for efficiency
                    custom_phrases.extend(self._extract_english_key_phrases(doc_text, top_n=10))

                # Count frequency of custom phrases
                custom_phrase_counts = Counter(custom_phrases)
                top_custom_phrases = custom_phrase_counts.most_common(50)

                # Use NLTK to extract collocations (bigrams and trigrams)
                all_words = []
                for doc in en_docs:
                    all_words.extend(self._preprocess_english_text(doc))

                # Extract bigrams
                bigram_measures = BigramAssocMeasures()
                bigram_finder = BigramCollocationFinder.from_words(all_words)
                # Apply frequency filter
                bigram_finder.apply_freq_filter(min_doc_frequency)
                top_bigrams = bigram_finder.nbest(bigram_measures.pmi, 30)
                bigram_phrases = [' '.join(bigram) for bigram in top_bigrams]

                # Extract trigrams
                trigram_measures = TrigramAssocMeasures()
                trigram_finder = TrigramCollocationFinder.from_words(all_words)
                trigram_finder.apply_freq_filter(min_doc_frequency // 2)  # Lower threshold for trigrams
                top_trigrams = trigram_finder.nbest(trigram_measures.pmi, 20)
                trigram_phrases = [' '.join(trigram) for trigram in top_trigrams]

                # Combine results
                en_keyphrases = {
                    'tfidf_terms': [(term, float(score)) for term, score in tfidf_scores[:50]],
                    'spacy_phrases': [(phrase, count) for phrase, count in top_custom_phrases],
                    'nltk_bigrams': bigram_phrases,
                    'nltk_trigrams': trigram_phrases,
                    'document_count': len(en_docs)
                }
            except Exception as e:
                print(f"Error processing English documents: {e}")
                en_keyphrases = {'error': str(e)}

        # Process Chinese documents
        zh_keyphrases = {}
        if zh_docs and CHINESE_SUPPORT:
            print(f"Extracting key phrases from {len(zh_docs)} Chinese documents...")

            # Extract key phrases using jieba
            all_zh_phrases = []
            for doc_text in zh_docs:
                phrases = self._extract_chinese_key_phrases(doc_text, top_n=15)
                all_zh_phrases.extend(phrases)

            # Count phrase frequencies
            zh_phrase_counts = Counter(all_zh_phrases)

            # Filter by minimum document frequency
            # (This is approximate since we're not tracking exact document frequency)
            min_count = max(min_doc_frequency, len(zh_docs) // 10)  # Adaptive threshold
            filtered_zh_phrases = {phrase: count for phrase, count in zh_phrase_counts.items()
                                   if count >= min_count}

            # Get top phrases
            top_zh_phrases = sorted(filtered_zh_phrases.items(),
                                    key=lambda x: x[1], reverse=True)[:50]

            zh_keyphrases = {
                'jieba_phrases': [(phrase, count) for phrase, count in top_zh_phrases],
                'document_count': len(zh_docs)
            }

    def analyze_all_themes(self) -> Dict[str, Any]:
        """Perform comprehensive thematic analysis using multiple methods with language support."""
        results = {}

        print("Starting comprehensive multilingual thematic analysis...")
        overall_start = time.time()

        # Get preprocessed documents for language statistics
        documents = self._preprocess_documents()

        # Get language distribution
        if documents:
            language_counts = Counter([doc.get('language', 'en') for doc in documents])
            print(f"Document language distribution: {dict(language_counts)}")
            results['language_distribution'] = dict(language_counts)

        # Apply different analysis methods
        try:
            print("\nAnalyzing thematic topics with NMF...")
            start = time.time()
            results['nmf_themes'] = self.identify_themes_with_nmf()
            print(f"✓ NMF analysis complete ({time.time() - start:.2f}s)")
        except Exception as e:
            self.logger.error(f"Error in NMF theme analysis: {str(e)}")
            results['nmf_themes'] = {"error": str(e)}
            print(f"✗ Error in NMF analysis: {str(e)}")

        try:
            print("\nBuilding concept network...")
            start = time.time()
            results['concept_network'] = self.identify_concept_network()
            print(f"✓ Concept network analysis complete ({time.time() - start:.2f}s)")
        except Exception as e:
            self.logger.error(f"Error in concept network analysis: {str(e)}")
            results['concept_network'] = {"error": str(e)}
            print(f"✗ Error in concept network analysis: {str(e)}")

        try:
            print("\nExtracting common key phrases...")
            start = time.time()
            results['key_phrases'] = self.extract_common_keyphrases()
            print(f"✓ Key phrase extraction complete ({time.time() - start:.2f}s)")
        except Exception as e:
            self.logger.error(f"Error in key phrase extraction: {str(e)}")
            results['key_phrases'] = {"error": str(e)}
            print(f"✗ Error in key phrase extraction: {str(e)}")

        print(f"\nThematic analysis completed in {time.time() - overall_start:.2f}s")

        return results

    def identify_themes_with_nmf(self, n_themes: int = 10) -> Dict[str, Any]:
        """Identify themes using Non-Negative Matrix Factorization (NMF) with language support."""
        documents = self._preprocess_documents()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Extract text content
        doc_texts = [doc['text'] for doc in documents]
        doc_metadata = [doc.get('metadata', {}) for doc in documents]
        doc_languages = [doc.get('language', 'en') for doc in documents]

        # Get document filenames if available
        doc_names = []
        for meta in doc_metadata:
            if 'file_name' in meta:
                doc_names.append(meta['file_name'])
            elif 'file_path' in meta:
                doc_names.append(Path(meta['file_path']).name)
            else:
                doc_names.append("Unknown document")

        # Create TF-IDF representation
        vectorizer = TfidfVectorizer(
            max_features=5000,
            min_df=2,
            max_df=0.85,
            stop_words='english'  # English stopwords handled by vectorizer
        )
        tfidf_matrix = vectorizer.fit_transform(doc_texts)

        # Apply NMF
        nmf_model = NMF(n_components=n_themes, random_state=42)
        nmf_features = nmf_model.fit_transform(tfidf_matrix)

        # Get top terms for each theme
        feature_names = vectorizer.get_feature_names_out()
        themes = {}

        for theme_idx, theme in enumerate(nmf_model.components_):
            # Get top terms for this theme
            top_term_indices = theme.argsort()[:-11:-1]  # Get indices of top 10 terms
            top_terms = [feature_names[i] for i in top_term_indices]

            # Get documents most associated with this theme
            theme_doc_scores = [(doc_idx, score) for doc_idx, score in enumerate(nmf_features[:, theme_idx])]
            theme_doc_scores.sort(key=lambda x: x[1], reverse=True)
            top_docs = [
                {
                    'name': doc_names[doc_idx],
                    'score': float(score),
                    'language': doc_languages[doc_idx]
                }
                for doc_idx, score in theme_doc_scores[:5] if score > 0.1
            ]

            themes[f"Theme {theme_idx + 1}"] = {
                'terms': top_terms,
                'top_documents': top_docs,
                'prevalence': float(np.sum(nmf_features[:, theme_idx]) / np.sum(nmf_features))
            }

        return {
            'method': 'Non-Negative Matrix Factorization',
            'num_themes': n_themes,
            'num_documents': len(documents),
            'language_distribution': Counter(doc_languages),
            'themes': themes
        }

    def identify_concept_network(self, min_co_occurrence: int = 3) -> Dict[str, Any]:
        """Build a network of co-occurring concepts across documents with language support."""
        documents = self._preprocess_documents()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Process documents to extract key terms based on language
        doc_terms = []
        doc_languages = []

        for doc in documents:
            language = doc.get('language', 'en')
            doc_languages.append(language)

            # Use appropriate preprocessing for each language
            if language == 'zh':
                terms = self._preprocess_chinese_text(doc['text'])
            else:
                terms = self._preprocess_english_text(doc['text'])

            doc_terms.append(terms)

        # Build term counter - separate by language for better analysis
        en_term_counter = Counter()
        zh_term_counter = Counter()

        for terms, language in zip(doc_terms, doc_languages):
            if language == 'zh':
                zh_term_counter.update(terms)
            else:
                en_term_counter.update(terms)

        # Filter to keep only common terms
        common_en_terms = [term for term, count in en_term_counter.items()
                           if count >= min_co_occurrence]
        common_zh_terms = [term for term, count in zh_term_counter.items()
                           if count >= min_co_occurrence]

        # Combine counters for graph building
        term_counter = en_term_counter + zh_term_counter
        common_terms = common_en_terms + common_zh_terms

        # Build co-occurrence network
        G = nx.Graph()

        # Add nodes for common terms
        for term in common_terms:
            G.add_node(term, frequency=term_counter[term])

        # Add edges for co-occurrences
        for terms in doc_terms:
            # Filter to common terms only
            doc_common_terms = [t for t in terms if t in common_terms]

            # Add edges between all pairs of terms in this document
            for i, term1 in enumerate(doc_common_terms):
                for term2 in doc_common_terms[i + 1:]:
                    if G.has_edge(term1, term2):
                        G[term1][term2]['weight'] += 1
                    else:
                        G.add_edge(term1, term2, weight=1)

        # Remove weak connections
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                           if d['weight'] < min_co_occurrence]
        G.remove_edges_from(edges_to_remove)

        # Find communities using Louvain method
        try:
            # First try the python-louvain package (community-detection)
            try:
                import community as community_louvain
                partition = community_louvain.best_partition(G)

                # Group concepts by community
                communities = {}
                for node, community_id in partition.items():
                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(node)

                # Keep only significant communities
                significant_communities = {
                    f"Concept Group {i + 1}": sorted(nodes)
                    for i, (_, nodes) in enumerate(
                        sorted(
                            [(k, v) for k, v in communities.items() if len(v) >= 3],
                            key=lambda x: len(x[1]),
                            reverse=True
                        )
                    )
                }
            except ImportError:
                # If community package not installed, use connected components instead
                significant_communities = {}
                for i, component in enumerate(nx.connected_components(G)):
                    if len(component) >= 3:  # Only keep significant components
                        significant_communities[f"Concept Group {i + 1}"] = sorted(component)

            # Calculate centrality measures
            centrality = nx.degree_centrality(G)

            # Get top concepts by centrality
            top_concepts = sorted(
                [(node, score) for node, score in centrality.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]

            # Separate top concepts by language
            en_top_concepts = []
            zh_top_concepts = []

            for node, score in top_concepts:
                # Check if the node contains Chinese characters
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in node)
                if has_chinese:
                    zh_top_concepts.append({"concept": node, "centrality": float(score)})
                else:
                    en_top_concepts.append({"concept": node, "centrality": float(score)})

            return {
                'method': 'Concept Co-occurrence Network',
                'num_documents': len(documents),
                'num_concepts': len(common_terms),
                'language_distribution': Counter(doc_languages),
                'english_concepts': len(common_en_terms),
                'chinese_concepts': len(common_zh_terms),
                'top_english_concepts': en_top_concepts,
                'top_chinese_concepts': zh_top_concepts,
                'top_concepts': [{"concept": node, "centrality": float(score)}
                                 for node, score in top_concepts],
                'concept_groups': significant_communities,
                'network_stats': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges(),
                    'communities': len(significant_communities)
                }
            }

        except ImportError:
            # If community detection package not available
            # Return basic network stats and top concepts
            centrality = nx.degree_centrality(G)
            top_concepts = sorted(
                [(node, score) for node, score in centrality.items()],
                key=lambda x: x[1],
                reverse=True
            )[:20]

            # Separate top concepts by language
            en_top_concepts = []
            zh_top_concepts = []

            for node, score in top_concepts:
                # Check if the node contains Chinese characters
                has_chinese = any('\u4e00' <= char <= '\u9fff' for char in node)
                if has_chinese:
                    zh_top_concepts.append({"concept": node, "centrality": float(score)})
                else:
                    en_top_concepts.append({"concept": node, "centrality": float(score)})

            return {
                'method': 'Concept Co-occurrence Network (without community detection)',
                'num_documents': len(documents),
                'num_concepts': len(common_terms),
                'language_distribution': Counter(doc_languages),
                'english_concepts': len(common_en_terms),
                'chinese_concepts': len(common_zh_terms),
                'top_english_concepts': en_top_concepts,
                'top_chinese_concepts': zh_top_concepts,
                'top_concepts': [{"concept": node, "centrality": float(score)}
                                 for node, score in top_concepts],
                'network_stats': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
            }

    def extract_common_keyphrases(self, min_doc_frequency: int = 2) -> Dict[str, Any]:
        """Extract common key phrases across all documents with language support."""
        documents = self._preprocess_documents()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Separate documents by language
        en_documents = []
        zh_documents = []
        other_documents = []

        for doc in documents:
            language = doc.get('language', 'en')
            if language == 'en':
                en_documents.append(doc)
            elif language == 'zh':
                zh_documents.append(doc)
            else:
                other_documents.append(doc)

        print(f"Documents by language: English: {len(en_documents)}, "
              f"Chinese: {len(zh_documents)}, Other: {len(other_documents)}")

        # Process documents by language
        en_phrases = self._extract_keyphrases_for_language(en_documents, 'en', min_doc_frequency)
        zh_phrases = self._extract_keyphrases_for_language(zh_documents, 'zh', min_doc_frequency)

        # Combine results
        return {
            'method': 'Multilingual Common Key Phrases',
            'num_documents': len(documents),
            'language_distribution': {
                'english': len(en_documents),
                'chinese': len(zh_documents),
                'other': len(other_documents)
            },
            'english_phrases': en_phrases,
            'chinese_phrases': zh_phrases
        }

    def _extract_keyphrases_for_language(self, documents, language, min_doc_frequency):
        """Extract key phrases for a specific language subset of documents."""
        if not documents:
            return []

        # Extract phrases from each document
        doc_phrases = []
        for doc in documents:
            # Use language-specific method
            phrases = self._extract_key_phrases(doc['text'], language=language)
            doc_phrases.append(phrases)

        # Count phrase occurrences across documents
        phrase_doc_count = Counter()

        # Count number of documents containing each phrase
        all_phrases = set()
        for phrases in doc_phrases:
            # Count each phrase only once per document
            unique_phrases = set(phrases)
            all_phrases.update(unique_phrases)
            for phrase in unique_phrases:
                phrase_doc_count[phrase] += 1

        # Filter to phrases that appear in multiple documents
        common_phrases = {
            phrase: count
            for phrase, count in phrase_doc_count.items()
            if count >= min_doc_frequency
        }

        # Calculate percentage of documents containing each phrase
        doc_count = len(documents)
        if doc_count == 0:
            return []

        phrase_coverage = {
            phrase: (count / doc_count) * 100
            for phrase, count in common_phrases.items()
        }

        # Sort phrases by document frequency
        sorted_phrases = sorted(
            phrase_coverage.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Return top phrases
        return [
            {
                'phrase': phrase,
                'doc_count': phrase_doc_count[phrase],
                'coverage_percentage': float(coverage)
            }
            for phrase, coverage in sorted_phrases[:30]  # Top 30 phrases
        ]

    def print_analysis_results(self, results: Dict[str, Any]) -> None:
        """Print thematic analysis results in a formatted way."""
        print("\n===== COMPREHENSIVE THEMATIC ANALYSIS =====\n")

        # Print language distribution
        if 'language_distribution' in results:
            print("\n=== Document Language Distribution ===")
            total_docs = sum(results['language_distribution'].values())
            for lang, count in results['language_distribution'].items():
                percentage = (count / total_docs) * 100 if total_docs > 0 else 0
                print(f"{lang}: {count} documents ({percentage:.1f}%)")

        # Print NMF themes
        if 'nmf_themes' in results and 'themes' in results['nmf_themes']:
            print("\n=== NMF Thematic Analysis ===")
            for theme_name, theme_data in results['nmf_themes']['themes'].items():
                print(f"\n{theme_name}:")
                print(f"Top terms: {', '.join(theme_data['terms'][:10])}")
                print(f"Prevalence: {theme_data['prevalence'] * 100:.1f}%")

        # Print concept network results
        if 'concept_network' in results:
            cn_results = results['concept_network']
            print("\n=== Concept Network Analysis ===")

            # Print English concepts
            if 'top_english_concepts' in cn_results and cn_results['top_english_concepts']:
                print("\nTop English Concepts:")
                for i, concept in enumerate(cn_results['top_english_concepts'][:10], 1):
                    print(f"{i}. {concept['concept']} (centrality: {concept['centrality']:.3f})")

            # Print Chinese concepts
            if 'top_chinese_concepts' in cn_results and cn_results['top_chinese_concepts']:
                print("\nTop Chinese Concepts:")
                for i, concept in enumerate(cn_results['top_chinese_concepts'][:10], 1):
                    print(f"{i}. {concept['concept']} (centrality: {concept['centrality']:.3f})")

        # Print key phrases
        if 'key_phrases' in results:
            kp_results = results['key_phrases']
            print("\n=== Key Phrases Analysis ===")

            # Print English phrases
            if 'english_phrases' in kp_results and kp_results['english_phrases']:
                print("\nTop English Phrases:")
                for i, phrase_data in enumerate(kp_results['english_phrases'][:10], 1):
                    print(f"{i}. {phrase_data['phrase']} "
                          f"(in {phrase_data['doc_count']} docs, "
                          f"{phrase_data['coverage_percentage']:.1f}% coverage)")

            # Print Chinese phrases
            if 'chinese_phrases' in kp_results and kp_results['chinese_phrases']:
                print("\nTop Chinese Phrases:")
                for i, phrase_data in enumerate(kp_results['chinese_phrases'][:10], 1):
                    print(f"{i}. {phrase_data['phrase']} "
                          f"(in {phrase_data['doc_count']} docs, "
                          f"{phrase_data['coverage_percentage']:.1f}% coverage)")

        print("\n===== ANALYSIS COMPLETE =====\n")