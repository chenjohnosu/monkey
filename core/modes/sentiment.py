"""
Sentiment analysis module with support for both English and Chinese text
"""

import os
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
from core.engine.logging import debug,warning,info
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.language.tokenizer import JIEBA_AVAILABLE, get_jieba_instance

# Try to import sentiment analysis libraries - use what's available
try:
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warning("transformers not available, falling back to simple lexicon-based sentiment analysis")

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk

    # Ensure NLTK data is downloaded
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except LookupError:
        info("Downloading NLTK VADER lexicon...")
        nltk.download('vader_lexicon', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    warning("NLTK not available, falling back to transformers or lexicon-based sentiment analysis")

class SentimentAnalyzer:
    """Analyzes sentiment in document content with support for English and Chinese"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the sentiment analyzer"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)

        # Initialize LLM connector factory for summarization
        try:
            from core.connectors.connector_factory import ConnectorFactory
            self.factory = ConnectorFactory(config)
            self.llm_connector = self.factory.get_llm_connector()
            debug(config, "LLM connector factory initialized for sentiment analysis")
        except Exception as e:
            debug(config, f"Error initializing ConnectorFactory: {str(e)}")
            self.factory = None
            self.llm_connector = None

        # Load Chinese sentiment lexicon if available
        self.chinese_sentiment_lexicon = self._load_chinese_sentiment_lexicon()

        # Initialize transformers sentiment analyzer if available
        self.en_sentiment_analyzer = None
        self.zh_sentiment_analyzer = None

        if TRANSFORMERS_AVAILABLE:
            try:
                # Use a smaller model for English sentiment
                self.en_sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english",
                    truncation=True
                )
                debug(config, "Initialized English sentiment analyzer with transformers")
            except Exception as e:
                debug(config, f"Error initializing English sentiment analyzer: {str(e)}")

        # Initialize NLTK VADER sentiment analyzer for English if available
        self.vader_analyzer = None
        if NLTK_AVAILABLE:
            try:
                self.vader_analyzer = SentimentIntensityAnalyzer()
                debug(config, "Initialized VADER sentiment analyzer")
            except Exception as e:
                debug(config, f"Error initializing VADER sentiment analyzer: {str(e)}")

        debug(config, "Sentiment analyzer initialized")

    def _load_chinese_sentiment_lexicon(self):
        """Load Chinese sentiment lexicon from file"""
        lexicon = {
            'positive': set(),
            'negative': set(),
            'intensity': {}
        }

        # Define lexicon directory
        lexicon_dir = 'lexicon'
        if not os.path.exists(lexicon_dir):
            os.makedirs(lexicon_dir)
            debug(self.config, f"Created lexicon directory: {lexicon_dir}")

        # Try to load custom Chinese sentiment lexicon if available
        try:
            # Check for positive words
            positive_path = os.path.join(lexicon_dir, 'chinese_positive.txt')
            if os.path.exists(positive_path):
                with open(positive_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        word = line.strip()
                        if word:
                            lexicon['positive'].add(word)
                            lexicon['intensity'][word] = 1.0
                debug(self.config, f"Loaded {len(lexicon['positive'])} Chinese positive sentiment words")

            # Check for negative words
            negative_path = os.path.join(lexicon_dir, 'chinese_negative.txt')
            if os.path.exists(negative_path):
                with open(negative_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        word = line.strip()
                        if word:
                            lexicon['negative'].add(word)
                            lexicon['intensity'][word] = -1.0
                debug(self.config, f"Loaded {len(lexicon['negative'])} Chinese negative sentiment words")

            # If no custom lexicon, use a small built-in lexicon
            if not lexicon['positive'] and not lexicon['negative']:
                debug(self.config, "No Chinese sentiment lexicon found, using built-in minimal lexicon")
                # Basic positive words
                positive_words = ["好", "喜欢", "棒", "优秀", "满意", "快乐", "赞", "支持", "成功", "高兴"]
                for word in positive_words:
                    lexicon['positive'].add(word)
                    lexicon['intensity'][word] = 1.0

                # Basic negative words
                negative_words = ["不", "坏", "差", "糟", "失败", "问题", "难", "担忧", "困难", "遗憾"]
                for word in negative_words:
                    lexicon['negative'].add(word)
                    lexicon['intensity'][word] = -1.0
        except Exception as e:
            debug(self.config, f"Error loading Chinese sentiment lexicon: {str(e)}")
            # Set up minimal default lexicon
            lexicon['positive'] = set(["好", "喜欢", "优秀", "满意"])
            lexicon['negative'] = set(["不", "坏", "差", "失败"])
            for word in lexicon['positive']:
                lexicon['intensity'][word] = 1.0
            for word in lexicon['negative']:
                lexicon['intensity'][word] = -1.0

        return lexicon

    def analyze(self, workspace, method='all'):
        """
        Perform sentiment analysis on documents in a workspace

        Args:
            workspace (str): The workspace to analyze
            method (str): Analysis method ('all', 'basic', 'advanced')
        """
        debug(self.config, f"Performing sentiment analysis on workspace '{workspace}' using method '{method}'")

        # Validate method
        valid_methods = ['all', 'basic', 'advanced']
        if method not in valid_methods:
            print(f"Invalid method: {method}. Must be one of: {', '.join(valid_methods)}")
            return

        # Check if workspace exists
        data_dir = os.path.join("data", workspace)
        if not os.path.exists(data_dir):
            print(f"Workspace '{workspace}' does not exist or has no documents")
            return

        # Get documents from storage
        docs = self.storage_manager.get_documents(workspace)
        if not docs:
            print(f"No documents found in workspace '{workspace}'")
            return

        print(f"Analyzing sentiment in {len(docs)} documents from workspace '{workspace}'")

        # Extract and preprocess document content
        doc_contents, doc_languages = self._extract_document_contents(docs)

        # Display language distribution
        self._display_language_stats(doc_languages)

        # Run selected analysis methods
        results = {}

        if method in ['all', 'basic']:
            print("\nRunning Basic Sentiment Analysis...")
            results['basic'] = self._analyze_basic_sentiment(doc_contents)

        if method in ['all', 'advanced']:
            print("\nRunning Advanced Sentiment Analysis...")
            results['advanced'] = self._analyze_advanced_sentiment(doc_contents)

        # Output results
        self._output_results(workspace, results, method)

        return results

    def _extract_document_contents(self, docs):
        """
        Extract content from documents and preprocess for analysis

        Args:
            docs (List[Dict]): Raw documents

        Returns:
            Tuple: (processed documents, language counts)
        """
        debug(self.config, "Extracting and preprocessing document content")

        doc_contents = []
        doc_languages = Counter()

        for doc in docs:
            # Extract content and metadata
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            language = doc.get("metadata", {}).get("language", "en")

            # Track language
            doc_languages[language] += 1

            # Skip empty documents
            if not content or len(content.strip()) == 0:
                debug(self.config, f"Skipping empty document: {source}")
                continue

            # Use raw content for sentiment analysis (better than processed content which removes stopwords)
            text_to_analyze = content

            # Add document to dataset
            doc_contents.append({
                "id": len(doc_contents),
                "source": source,
                "language": language,
                "content": text_to_analyze
            })

        return doc_contents, doc_languages

    def _display_language_stats(self, language_counts):
        """
        Display language distribution in the document set

        Args:
            language_counts (Counter): Counts of languages
        """
        print("\nDocument Language Distribution:")
        total = sum(language_counts.values())

        for lang, count in language_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {lang}: {count} documents ({percentage:.1f}%)")

    def _analyze_basic_sentiment(self, doc_contents):
        """
        Perform basic sentiment analysis using lexicon-based approach

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug(self.config, "Performing basic sentiment analysis")

        # Initialize results structure
        results = {
            "method": "Basic Sentiment Analysis",
            "document_count": len(doc_contents),
            "sentiment_distribution": {},
            "document_sentiments": []
        }

        # Process each document
        for doc in doc_contents:
            doc_id = doc["id"]
            language = doc["language"]
            text = doc["content"]
            source = doc["source"]

            # Analyze sentiment based on language
            if language == "zh":
                sentiment_score, sentiment_label = self._analyze_chinese_sentiment(text)
            else:
                sentiment_score, sentiment_label = self._analyze_english_sentiment(text)

            # Store document sentiment result
            doc_sentiment = {
                "source": source,
                "sentiment": sentiment_label,
                "score": sentiment_score,
                "language": language
            }

            results["document_sentiments"].append(doc_sentiment)

            # Update sentiment distribution
            if sentiment_label not in results["sentiment_distribution"]:
                results["sentiment_distribution"][sentiment_label] = 0
            results["sentiment_distribution"][sentiment_label] += 1

        # Calculate overall sentiment stats
        total_docs = len(doc_contents)
        overall_distribution = {}
        for label, count in results["sentiment_distribution"].items():
            overall_distribution[label] = round((count / total_docs) * 100, 1)

        results["overall_distribution"] = overall_distribution

        # Calculate most common sentiments by language
        sentiment_by_language = defaultdict(lambda: defaultdict(int))
        for doc in results["document_sentiments"]:
            lang = doc["language"]
            sentiment = doc["sentiment"]
            sentiment_by_language[lang][sentiment] += 1

        results["sentiment_by_language"] = {
            lang: dict(counts) for lang, counts in sentiment_by_language.items()
        }

        return results

    def _analyze_english_sentiment(self, text):
        """
        Analyze sentiment in English text

        Args:
            text (str): English text to analyze

        Returns:
            Tuple: (sentiment score, sentiment label)
        """
        # Use transformers if available
        if self.en_sentiment_analyzer:
            try:
                # Truncate if too long
                truncated_text = text[:1024]
                result = self.en_sentiment_analyzer(truncated_text)[0]
                label = result["label"]
                score = result["score"]

                # Normalize label and score
                if label == "POSITIVE":
                    sentiment_label = "positive"
                    sentiment_score = score
                elif label == "NEGATIVE":
                    sentiment_label = "negative"
                    sentiment_score = -score
                else:
                    sentiment_label = "neutral"
                    sentiment_score = 0.0

                return sentiment_score, sentiment_label
            except Exception as e:
                debug(self.config, f"Error in transformer sentiment analysis: {str(e)}")
                # Fall back to VADER or lexicon

        # Use VADER if available
        if self.vader_analyzer:
            try:
                # Truncate if too long
                truncated_text = text[:5000]  # VADER handles longer text than transformers
                scores = self.vader_analyzer.polarity_scores(truncated_text)
                compound_score = scores["compound"]

                # Determine sentiment label based on compound score
                if compound_score >= 0.05:
                    sentiment_label = "positive"
                elif compound_score <= -0.05:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                return compound_score, sentiment_label
            except Exception as e:
                debug(self.config, f"Error in VADER sentiment analysis: {str(e)}")
                # Fall back to lexicon

        # Use simple lexicon as fallback
        return self._analyze_with_simple_lexicon(text, "en")

    def _analyze_chinese_sentiment(self, text):
        """
        Analyze sentiment in Chinese text

        Args:
            text (str): Chinese text to analyze

        Returns:
            Tuple: (sentiment score, sentiment label)
        """
        # Use transformers if available and Chinese sentiment model is loaded
        if self.zh_sentiment_analyzer:
            try:
                # Truncate if too long
                truncated_text = text[:512]  # Chinese characters may need shorter text
                result = self.zh_sentiment_analyzer(truncated_text)[0]
                label = result["label"]
                score = result["score"]

                # Normalize label
                if "正面" in label or "积极" in label or "positive" in label.lower():
                    sentiment_label = "positive"
                    sentiment_score = score
                elif "负面" in label or "消极" in label or "negative" in label.lower():
                    sentiment_label = "negative"
                    sentiment_score = -score
                else:
                    sentiment_label = "neutral"
                    sentiment_score = 0.0

                return sentiment_score, sentiment_label
            except Exception as e:
                debug(self.config, f"Error in Chinese transformer sentiment analysis: {str(e)}")
                # Fall back to lexicon

        # Use lexicon-based method as primary approach for Chinese
        return self._analyze_with_chinese_lexicon(text)

    def _analyze_with_chinese_lexicon(self, text):
        """
        Analyze Chinese sentiment using lexicon-based approach

        Args:
            text (str): Chinese text to analyze

        Returns:
            Tuple: (sentiment score, sentiment label)
        """
        # Segment text if jieba is available
        if JIEBA_AVAILABLE:
            jieba_instance = get_jieba_instance()
            if jieba_instance:
                words = list(jieba_instance.cut(text))
            else:
                # Character-based approach as fallback
                words = [char for char in text]
        else:
            # Character-based approach as fallback
            words = [char for char in text]

        # Count positive and negative words
        positive_count = 0
        negative_count = 0
        total_score = 0.0

        for word in words:
            if word in self.chinese_sentiment_lexicon['positive']:
                positive_count += 1
                total_score += self.chinese_sentiment_lexicon['intensity'].get(word, 1.0)
            elif word in self.chinese_sentiment_lexicon['negative']:
                negative_count += 1
                total_score += self.chinese_sentiment_lexicon['intensity'].get(word, -1.0)

        # Check for negation words that invert sentiment
        negation_words = ["不", "没", "无", "非", "莫", "勿", "未", "别", "甭", "休"]
        for i, word in enumerate(words):
            if word in negation_words and i + 1 < len(words):
                next_word = words[i + 1]
                if next_word in self.chinese_sentiment_lexicon['positive']:
                    positive_count -= 1
                    negative_count += 1
                    total_score -= 2 * self.chinese_sentiment_lexicon['intensity'].get(next_word, 1.0)
                elif next_word in self.chinese_sentiment_lexicon['negative']:
                    negative_count -= 1
                    positive_count += 1
                    total_score -= 2 * self.chinese_sentiment_lexicon['intensity'].get(next_word, -1.0)

        # Normalize score
        if len(words) > 0:
            normalized_score = total_score / len(words)
        else:
            normalized_score = 0.0

        # Determine sentiment label
        if normalized_score > 0.03:  # Small threshold for positive
            sentiment_label = "positive"
        elif normalized_score < -0.02:  # Smaller threshold for negative (negative words are often stronger)
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return normalized_score, sentiment_label

    def _analyze_with_simple_lexicon(self, text, language="en"):
        """
        Analyze sentiment using a simple lexicon-based approach (fallback method)

        Args:
            text (str): Text to analyze
            language (str): Language code

        Returns:
            Tuple: (sentiment score, sentiment label)
        """
        # Simple English sentiment lexicon
        en_positive = {"good", "great", "excellent", "positive", "wonderful", "happy", "pleased", "satisfied",
                       "success", "successful", "best", "better", "benefit", "effective", "recommend"}
        en_negative = {"bad", "poor", "terrible", "negative", "unhappy", "disappointed", "disappointing",
                       "fail", "failure", "worst", "worse", "problem", "difficult", "hard", "complaint"}

        # Convert to lowercase
        text = text.lower()

        # Count positive and negative words
        positive_count = sum(1 for word in en_positive if word in text)
        negative_count = sum(1 for word in en_negative if word in text)

        # Calculate simple sentiment score
        total_count = positive_count + negative_count
        if total_count > 0:
            sentiment_score = (positive_count - negative_count) / total_count
        else:
            sentiment_score = 0.0

        # Determine sentiment label
        if sentiment_score > 0.1:
            sentiment_label = "positive"
        elif sentiment_score < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"

        return sentiment_score, sentiment_label

    def _analyze_advanced_sentiment(self, doc_contents):
        """
        Perform advanced sentiment analysis with aspect extraction and summarization

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug(self.config, "Performing advanced sentiment analysis")

        # Group documents by language
        docs_by_language = defaultdict(list)

        for doc in doc_contents:
            language = doc["language"]
            docs_by_language[language].append(doc)

        # Initialize results
        results = {
            "method": "Advanced Sentiment Analysis",
            "document_count": len(doc_contents),
            "languages": list(docs_by_language.keys()),
            "sentiment_trends": [],
            "document_analysis": []
        }

        # Process each document for detailed analysis
        for doc in doc_contents:
            doc_id = doc["id"]
            language = doc["language"]
            text = doc["content"]
            source = doc["source"]

            # Extract aspects and their sentiment
            aspects = self._extract_sentiment_aspects(text, language)

            # Analyze overall sentiment
            sentiment_score, sentiment_label = self._analyze_document_sentiment(text, language)

            # Identify emotional tone
            emotional_tone = self._identify_emotional_tone(text, language)

            # Generate interpretive summary using LLM if available
            summary = self._generate_sentiment_summary(text, aspects, sentiment_label, emotional_tone)

            # Store document analysis
            doc_analysis = {
                "source": source,
                "overall_sentiment": sentiment_label,
                "sentiment_score": sentiment_score,
                "emotional_tone": emotional_tone,
                "aspects": aspects,
                "summary": summary
            }

            results["document_analysis"].append(doc_analysis)

        # Identify sentiment trends across documents
        trends = self._identify_sentiment_trends(results["document_analysis"])
        results["sentiment_trends"] = trends

        return results

    def _analyze_document_sentiment(self, text, language):
        """
        Analyze overall sentiment in a document with more nuanced scoring

        Args:
            text (str): Text to analyze
            language (str): Language code

        Returns:
            Tuple: (sentiment score, sentiment label)
        """
        # For Chinese documents
        if language == "zh":
            return self._analyze_chinese_sentiment(text)

        # For English documents - try transformer model first, VADER second, then lexicon
        if self.en_sentiment_analyzer:
            try:
                # Analyze chunks of text and average results
                chunks = self._split_text_into_chunks(text, 512)
                scores = []

                for chunk in chunks:
                    if not chunk or len(chunk.strip()) == 0:
                        continue

                    result = self.en_sentiment_analyzer(chunk)[0]
                    label = result["label"]
                    score = result["score"]

                    # Convert to numeric score
                    if label == "POSITIVE":
                        scores.append(score)
                    elif label == "NEGATIVE":
                        scores.append(-score)
                    else:
                        scores.append(0.0)

                # Average scores
                if scores:
                    avg_score = sum(scores) / len(scores)
                else:
                    avg_score = 0.0

                # Determine sentiment label
                if avg_score >= 0.2:
                    sentiment_label = "positive"
                elif avg_score <= -0.2:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                return avg_score, sentiment_label
            except Exception as e:
                debug(self.config, f"Error in transformer sentiment analysis: {str(e)}")

        # Try VADER if available
        if self.vader_analyzer:
            try:
                # Analyze chunks of text and average results
                chunks = self._split_text_into_chunks(text, 1000)
                scores = []

                for chunk in chunks:
                    if not chunk or len(chunk.strip()) == 0:
                        continue

                    result = self.vader_analyzer.polarity_scores(chunk)
                    scores.append(result["compound"])

                # Average scores
                if scores:
                    avg_score = sum(scores) / len(scores)
                else:
                    avg_score = 0.0

                # Determine sentiment label
                if avg_score >= 0.05:
                    sentiment_label = "positive"
                elif avg_score <= -0.05:
                    sentiment_label = "negative"
                else:
                    sentiment_label = "neutral"

                return avg_score, sentiment_label
            except Exception as e:
                debug(self.config, f"Error in VADER sentiment analysis: {str(e)}")

        # Fall back to lexicon approach
        return self._analyze_with_simple_lexicon(text, language)

    def _split_text_into_chunks(self, text, chunk_size=500):
        """
        Split text into chunks for processing

        Args:
            text (str): Text to split
            chunk_size (int): Maximum chunk size

        Returns:
            List[str]: Text chunks
        """
        # Split by sentences first if possible
        chunks = []
        sentences = text.split('.')

        current_chunk = ""

        for sentence in sentences:
            sentence = sentence.strip() + "."

            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # If sentence is longer than chunk_size, split it
                if len(sentence) > chunk_size:
                    words = sentence.split()
                    current_chunk = ""

                    for word in words:
                        if len(current_chunk) + len(word) + 1 <= chunk_size:
                            current_chunk += " " + word
                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = word
                else:
                    current_chunk = sentence

        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _extract_sentiment_aspects(self, text, language):
        """
        Extract aspects and their associated sentiment

        Args:
            text (str): Text to analyze
            language (str): Language code

        Returns:
            List[Dict]: Extracted aspects with sentiment
        """
        aspects = []

        # Use LLM to extract aspects if available
        if self.llm_connector:
            try:
                return self._extract_aspects_with_llm(text, language)
            except Exception as e:
                debug(self.config, f"Error extracting aspects with LLM: {str(e)}")

        # Otherwise use rule-based extraction (simplified)
        if language == "zh":
            # Extract aspects from Chinese text
            return self._extract_chinese_aspects(text)
        else:
            # Extract aspects from English text
            return self._extract_english_aspects(text)

    def _extract_aspects_with_llm(self, text, language):
        """
        Extract sentiment aspects using an LLM

        Args:
            text (str): Text to analyze
            language (str): Language code

        Returns:
            List[Dict]: Extracted aspects with sentiment
        """
        # Create prompt based on language
        if language == "zh":
            prompt = f"""请从以下文本中提取主要情感方面及其情感极性。
文本: {text[:2000]}...
请识别3-5个主要方面并标注其情感（积极、消极或中性）。
格式如下:
方面1: [方面名称], 情感: [积极/消极/中性], 置信度: [0-1]
方面2: [方面名称], 情感: [积极/消极/中性], 置信度: [0-1]
只输出方面列表，不需要其他解释。"""
        else:
            prompt = f"""Extract the main sentiment aspects from the following text.
Text: {text[:2000]}...
Identify 3-5 main aspects and note their sentiment (positive, negative, or neutral).
Format as follows:
Aspect 1: [aspect name], Sentiment: [positive/negative/neutral], Confidence: [0-1]
Aspect 2: [aspect name], Sentiment: [positive/negative/neutral], Confidence: [0-1]
Output only the list of aspects without any other explanation."""

        # Generate response
        model = self.config.get('llm.default_model')
        try:
            response = self.llm_connector.generate(prompt, model=model, max_tokens=500)

            # Parse the response
            aspects = []
            lines = response.strip().split('\n')

            for line in lines:
                if not line or ":" not in line:
                    continue

                try:
                    # Extract aspect name
                    if "Aspect" in line or "方面" in line:
                        parts = line.split(":", 2)
                        aspect_name = parts[1].split(",")[0].strip()
                    else:
                        continue

                    # Extract sentiment
                    if "Sentiment:" in line or "情感:" in line:
                        sentiment_part = line.split("Sentiment:")[1].split(",")[0].strip() if "Sentiment:" in line else \
                        line.split("情感:")[1].split(",")[0].strip()
                    else:
                        sentiment_part = ""

                    # Normalize sentiment
                    if sentiment_part.lower() in ["positive", "积极"]:
                        sentiment = "positive"
                    elif sentiment_part.lower() in ["negative", "消极"]:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                    # Extract confidence if available
                    confidence = 0.7  # Default
                    if "Confidence:" in line or "置信度:" in line:
                        try:
                            conf_part = line.split("Confidence:")[1].strip() if "Confidence:" in line else \
                            line.split("置信度:")[1].strip()
                            confidence = float(conf_part)
                        except:
                            pass

                    aspects.append({
                        "aspect": aspect_name,
                        "sentiment": sentiment,
                        "confidence": confidence
                    })
                except Exception as e:
                    debug(self.config, f"Error parsing aspect line: {line}, {str(e)}")

            return aspects
        except Exception as e:
            debug(self.config, f"Error generating aspects with LLM: {str(e)}")
            return []

    def _extract_chinese_aspects(self, text):
        """
        Extract aspects from Chinese text using rule-based approach

        Args:
            text (str): Chinese text

        Returns:
            List[Dict]: Extracted aspects with sentiment
        """
        # Simplified rule-based approach
        aspects = []

        # Basic aspect categories for Chinese
        aspect_categories = ["质量", "服务", "价格", "体验", "功能", "设计", "使用", "效果"]

        # Check if jieba is available for word segmentation
        if JIEBA_AVAILABLE:
            words = list(jieba.cut(text))
        else:
            # Character-based approach as fallback
            words = [char for char in text]

        # Find sentences or segments containing aspect keywords
        sentences = text.split('。')

        for category in aspect_categories:
            # Check if the category appears in the text
            if category in text:
                # Find sentences containing this category
                relevant_sentences = [s for s in sentences if category in s]

                if relevant_sentences:
                    # Analyze sentiment for these sentences
                    sentiment_scores = []
                    for sentence in relevant_sentences:
                        score, _ = self._analyze_chinese_sentiment(sentence)
                        sentiment_scores.append(score)

                    # Average sentiment score
                    avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

                    # Determine sentiment
                    if avg_score > 0.05:
                        sentiment = "positive"
                    elif avg_score < -0.05:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                    aspects.append({
                        "aspect": category,
                        "sentiment": sentiment,
                        "confidence": min(abs(avg_score) + 0.5, 0.9)  # Confidence based on sentiment strength
                    })

        return aspects

    def _extract_english_aspects(self, text):
        """
        Extract aspects from English text using rule-based approach

        Args:
            text (str): English text

        Returns:
            List[Dict]: Extracted aspects with sentiment
        """
        # Common aspect categories for general document analysis
        aspect_categories = ["quality", "performance", "features", "design", "usability",
                             "reliability", "efficiency", "value", "implementation", "results"]

        aspects = []

        # Split into sentences
        sentences = text.split('.')

        for category in aspect_categories:
            # Check if the category appears in the text
            if category in text.lower():
                # Find sentences containing this category
                relevant_sentences = [s for s in sentences if category in s.lower()]

                if relevant_sentences:
                    # Analyze sentiment for these sentences
                    sentiment_scores = []

                    # Use VADER if available for sentence-level analysis
                    if self.vader_analyzer:
                        for sentence in relevant_sentences:
                            scores = self.vader_analyzer.polarity_scores(sentence)
                            sentiment_scores.append(scores["compound"])
                    else:
                        # Fallback to simple lexicon
                        for sentence in relevant_sentences:
                            score, _ = self._analyze_with_simple_lexicon(sentence)
                            sentiment_scores.append(score)

                    # Average sentiment score
                    avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

                    # Determine sentiment
                    if avg_score > 0.05:
                        sentiment = "positive"
                    elif avg_score < -0.05:
                        sentiment = "negative"
                    else:
                        sentiment = "neutral"

                    aspects.append({
                        "aspect": category,
                        "sentiment": sentiment,
                        "confidence": min(abs(avg_score) + 0.5, 0.9)  # Confidence based on sentiment strength
                    })

        return aspects

    def _identify_emotional_tone(self, text, language):
        """
        Identify emotional tone in the text

        Args:
            text (str): Text to analyze
            language (str): Language code

        Returns:
            str: Identified emotional tone
        """
        # Define emotion lexicons
        emotions = {
            "joy": {"happy", "pleased", "delighted", "excited", "satisfied", "glad", "thrilled",
                    "enthusiastic", "cheerful", "content"},
            "anger": {"angry", "annoyed", "irritated", "frustrated", "furious", "outraged",
                      "hostile", "mad", "upset", "enraged"},
            "sadness": {"sad", "unhappy", "disappointed", "depressed", "gloomy", "heartbroken",
                        "miserable", "melancholy", "sorrowful", "downcast"},
            "fear": {"afraid", "scared", "frightened", "terrified", "anxious", "worried",
                     "nervous", "concerned", "uneasy", "alarmed"},
            "surprise": {"surprised", "amazed", "astonished", "stunned", "shocked",
                         "startled", "unexpected", "remarkable", "extraordinary"}
        }

        # Chinese emotion lexicons
        zh_emotions = {
            "joy": {"快乐", "高兴", "开心", "兴奋", "满意", "愉快", "欣喜", "欢乐", "喜悦", "舒心"},
            "anger": {"生气", "愤怒", "恼火", "不满", "烦躁", "恼怒", "气愤", "暴怒", "气恼", "恼"},
            "sadness": {"悲伤", "难过", "失望", "伤心", "忧郁", "悲痛", "忧伤", "痛苦", "悲", "丧气"},
            "fear": {"害怕", "恐惧", "担忧", "忧虑", "惊恐", "惊慌", "紧张", "焦虑", "担心", "慌张"},
            "surprise": {"惊讶", "惊奇", "震惊", "意外", "吃惊", "惊讶", "惊愕", "诧异", "错愕", "惊异"}
        }

        # Lowercase the text for English
        if language != "zh":
            text = text.lower()

        # Count emotion words
        emotion_counts = {emotion: 0 for emotion in emotions}

        if language == "zh":
            # For Chinese text
            for emotion, words in zh_emotions.items():
                for word in words:
                    if word in text:
                        emotion_counts[emotion] += text.count(word)
        else:
            # For English text
            for emotion, words in emotions.items():
                for word in words:
                    if f" {word} " in f" {text} ":  # Add spaces to match whole words
                        emotion_counts[emotion] += 1

        # Find the dominant emotion
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])

        # If no emotions detected, return "neutral"
        if dominant_emotion[1] == 0:
            return "neutral"

        return dominant_emotion[0]

    def _identify_sentiment_trends(self, doc_analyses):
        """
        Identify sentiment trends across documents

        Args:
            doc_analyses (List[Dict]): Document analyses

        Returns:
            List[Dict]: Identified sentiment trends
        """
        # Group documents by sentiment
        sentiment_groups = defaultdict(list)
        for doc in doc_analyses:
            sentiment = doc["overall_sentiment"]
            sentiment_groups[sentiment].append(doc)

        # Track aspect sentiments across documents
        aspect_sentiments = defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0})

        for doc in doc_analyses:
            for aspect in doc.get("aspects", []):
                aspect_name = aspect["aspect"]
                sentiment = aspect["sentiment"]
                aspect_sentiments[aspect_name][sentiment] += 1

        # Identify dominant aspects and their sentiment trends
        trends = []

        for aspect, counts in aspect_sentiments.items():
            total = sum(counts.values())
            if total >= 2:  # Only include aspects that appear in at least 2 documents
                # Calculate percentage for each sentiment
                percentages = {
                    sentiment: (count / total) * 100
                    for sentiment, count in counts.items()
                }

                # Determine dominant sentiment
                dominant_sentiment = max(counts.items(), key=lambda x: x[1])[0]

                trends.append({
                    "aspect": aspect,
                    "dominant_sentiment": dominant_sentiment,
                    "document_count": total,
                    "sentiment_distribution": percentages
                })

        # Sort trends by document count
        trends.sort(key=lambda x: x["document_count"], reverse=True)

        return trends

    def _generate_sentiment_summary(self, text, aspects, sentiment, emotional_tone):
        """
        Generate an interpretive summary of sentiment analysis

        Args:
            text (str): Original text
            aspects (List[Dict]): Extracted aspects
            sentiment (str): Overall sentiment
            emotional_tone (str): Emotional tone

        Returns:
            str: Generated summary
        """
        # Use LLM for summarization if available
        if self.llm_connector:
            try:
                # Create a prompt for the LLM
                prompt = f"""Generate a short summary (2-3 sentences) of the sentiment analysis results.

Overall sentiment: {sentiment}
Emotional tone: {emotional_tone}
Key aspects: {', '.join([f"{a['aspect']} ({a['sentiment']})" for a in aspects])}

Text excerpt: {text[:500]}...

Please summarize the sentiment profile in 2-3 concise sentences, focusing on the emotional tone and key aspects."""

                # Generate response
                model = self.config.get('llm.default_model')
                summary = self.llm_connector.generate(prompt, model=model, max_tokens=200)

                return summary.strip()
            except Exception as e:
                debug(self.config, f"Error generating sentiment summary with LLM: {str(e)}")

        # Generate a template-based summary if LLM is not available
        try:
            # Get aspect information
            positive_aspects = [a["aspect"] for a in aspects if a["sentiment"] == "positive"]
            negative_aspects = [a["aspect"] for a in aspects if a["sentiment"] == "negative"]

            # Create summary based on overall sentiment
            if sentiment == "positive":
                summary = f"The document expresses a generally positive sentiment with a {emotional_tone} emotional tone."
                if positive_aspects:
                    summary += f" Positive aspects include {', '.join(positive_aspects[:2])}."
                if negative_aspects:
                    summary += f" Despite the overall positive tone, concerns about {', '.join(negative_aspects[:1])} were identified."
            elif sentiment == "negative":
                summary = f"The document conveys a negative sentiment with a {emotional_tone} emotional tone."
                if negative_aspects:
                    summary += f" Negative aspects include {', '.join(negative_aspects[:2])}."
                if positive_aspects:
                    summary += f" Some positive elements were noted regarding {', '.join(positive_aspects[:1])}."
            else:
                summary = f"The document has a neutral sentiment with a {emotional_tone} emotional tone."
                if positive_aspects or negative_aspects:
                    summary += f" The analysis revealed mixed opinions about {', '.join((positive_aspects + negative_aspects)[:2])}."

            return summary
        except Exception as e:
            debug(self.config, f"Error generating template summary: {str(e)}")
            return f"Document expresses {sentiment} sentiment with a {emotional_tone} emotional tone."

    def _output_results(self, workspace, results, method):
        """
        Output sentiment analysis results

        Args:
            workspace (str): Target workspace
            results (Dict): Analysis results
            method (str): Analysis method
        """
        debug(self.config, "Outputting sentiment analysis results")

        print("\nSentiment Analysis Results:")

        # Display results for each method
        for m, result in results.items():
            print(f"\n{result['method']}:")

            # Display sentiment distribution
            if "sentiment_distribution" in result:
                print("\n  Sentiment Distribution:")
                for sentiment, count in result["sentiment_distribution"].items():
                    print(f"    {sentiment.capitalize()}: {count} documents")

            # Display overall distribution percentages
            if "overall_distribution" in result:
                print("\n  Overall Distribution (%):")
                for sentiment, percentage in result["overall_distribution"].items():
                    print(f"    {sentiment.capitalize()}: {percentage}%")

            # Display sentiment by language
            if "sentiment_by_language" in result:
                print("\n  Sentiment by Language:")
                for language, sentiments in result["sentiment_by_language"].items():
                    print(f"    {language}:")
                    for sentiment, count in sentiments.items():
                        print(f"      {sentiment.capitalize()}: {count}")

            # Display individual document sentiments (limited to first 5)
            if "document_sentiments" in result:
                print("\n  Document Sentiments (first 5):")
                for i, doc in enumerate(result["document_sentiments"][:5]):
                    print(f"    {doc['source']}: {doc['sentiment']} (score: {doc['score']:.2f})")

                if len(result["document_sentiments"]) > 5:
                    print(f"    ... and {len(result['document_sentiments']) - 5} more")

            # Display sentiment trends
            if "sentiment_trends" in result:
                print("\n  Sentiment Trends (Across Documents):")
                for trend in result["sentiment_trends"]:
                    print(f"    Aspect: {trend['aspect']}")
                    print(f"      Dominant Sentiment: {trend['dominant_sentiment']}")
                    print(f"      Appears in {trend['document_count']} documents")
                    print(
                        f"      Distribution: {', '.join([f'{s.capitalize()} {p:.1f}%' for s, p in trend['sentiment_distribution'].items() if p > 0])}")

            # Display advanced document analysis (limited to first 3)
            if "document_analysis" in result:
                print("\n  Detailed Document Analysis (first 3):")
                for i, doc in enumerate(result["document_analysis"][:3]):
                    print(f"\n    Document: {doc['source']}")
                    print(f"      Overall Sentiment: {doc['overall_sentiment']}")
                    print(f"      Emotional Tone: {doc['emotional_tone']}")

                    if doc.get("aspects"):
                        print("      Key Aspects:")
                        for aspect in doc["aspects"]:
                            print(
                                f"        {aspect['aspect']}: {aspect['sentiment']} (confidence: {aspect['confidence']:.2f})")

                    if doc.get("summary"):
                        print(f"      Summary: {doc['summary']}")

                if len(result["document_analysis"]) > 3:
                    print(f"\n    ... and {len(result['document_analysis']) - 3} more documents")

        # Save results to file
        output_format = self.config.get('system.output_format')
        filepath = self.output_manager.save_sentiment_analysis(workspace, results, "sentiment", output_format)

        print(f"\nResults saved to: {filepath}")