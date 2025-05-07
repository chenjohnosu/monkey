"""
Keyword extraction methods for document analysis toolkit
Provides multiple extraction algorithms with language-specific handling
"""

import os
import re
from typing import List, Set, Dict, Any, Optional, Callable
import numpy as np
from collections import Counter

from core.engine.logging import debug, warning, info, error, trace
from core.engine.dependencies import require, is_available
from core.language.spacy_tokenizer import SPACY_AVAILABLE, get_spacy_model, load_stopwords


def extract_keywords(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using the configured method

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    # Get configured method
    method = config.get('keywords.method', 'spacy')

    # Log which method is being used
    debug(config, f"Extracting keywords using {method} method for {len(texts)} texts in {language}")

    # Load language-specific stopwords if needed
    if language == 'zh' and stopwords is None:
        stopwords = load_chinese_stopwords()

    # Call appropriate method
    if method == 'spacy':
        return extract_keywords_spacy(config, texts, language, top_n, stopwords)
    elif method == 'rake-nltk':
        return extract_keywords_rake(config, texts, language, top_n, stopwords)
    elif method == 'yake':
        return extract_keywords_yake(config, texts, language, top_n, stopwords)
    elif method == 'keybert':
        return extract_keywords_keybert(config, texts, language, top_n, stopwords)
    else:
        # Default to TF-IDF
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)


def extract_keywords_spacy(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using spaCy's linguistic features

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    if not SPACY_AVAILABLE:
        warning("spaCy not available for keyword extraction, falling back to TF-IDF")
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)

    debug(config, f"Extracting keywords with spaCy from {len(texts)} texts in {language}")

    if not texts:
        return []

    try:
        # Load appropriate spaCy model
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}, falling back to TF-IDF")
            return extract_keywords_tfidf(config, texts, language, top_n, stopwords)

        # Combine texts into a manageable chunk to process
        combined_text = " ".join(texts)
        if len(combined_text) > 100000:  # Limit text length to avoid memory issues
            combined_text = combined_text[:100000]

        # Process with spaCy
        doc = nlp(combined_text)

        # Count term frequencies considering various factors
        term_freqs = Counter()

        # Extract important terms based on POS tags
        for token in doc:
            # Skip stopwords, punctuation and very short words
            if token.is_stop or token.is_punct or len(token.text) < 2:
                continue

            # Focus on nouns, verbs, and adjectives as they often form meaningful keywords
            if token.pos_ in ('NOUN', 'PROPN', 'VERB', 'ADJ'):
                # Use lemma for normalization in non-Chinese languages
                if language != 'zh':
                    term = token.lemma_.lower()
                else:
                    term = token.text

                # Skip very short terms and custom stopwords
                if len(term) < 2 or (stopwords and term in stopwords):
                    continue

                # Weight terms by their part of speech (nouns are more important)
                weight = 1.0
                if token.pos_ in ('NOUN', 'PROPN'):
                    weight = 1.5
                elif token.pos_ == 'VERB':
                    weight = 1.0
                elif token.pos_ == 'ADJ':
                    weight = 0.8

                term_freqs[term] += weight

        # Extract named entities as they make good keywords
        for ent in doc.ents:
            if stopwords and ent.text.lower() in stopwords:
                continue
            term_freqs[ent.text] += 2.0  # Give entities higher weight

        # Extract noun phrases (for multi-word keywords)
        noun_phrases = []
        for chunk in doc.noun_chunks:
            # Clean the chunk text
            clean_chunk = ' '.join([token.text for token in chunk
                                    if not token.is_stop and not token.is_punct])
            if clean_chunk and len(clean_chunk) > 2:
                noun_phrases.append(clean_chunk)
                term_freqs[clean_chunk] += 1.5  # Give noun phrases good weight

        # Get the top keywords
        return [term for term, _ in term_freqs.most_common(top_n)]

    except Exception as e:
        debug(config, f"Error extracting keywords with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)


def extract_keywords_tfidf(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using TF-IDF

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    if not require('sklearn', 'keyword extraction with TF-IDF'):
        warning("scikit-learn not available for TF-IDF extraction")
        return []

    debug(config, f"Extracting keywords with TF-IDF from {len(texts)} texts in {language}")

    if not texts:
        return []

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Get vectorizer configured for corpus
        vectorizer = configure_vectorizer(config, len(texts), language, stopwords)
        if not vectorizer:
            return []

        # Create document-term matrix
        X = vectorizer.fit_transform(texts)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score for each term
        avg_scores = np.asarray(X.mean(axis=0)).ravel()

        # Sort terms by score
        scored_terms = list(zip(feature_names, avg_scores))
        sorted_terms = sorted(scored_terms, key=lambda x: x[1], reverse=True)

        # Filter stopwords if provided
        if stopwords:
            sorted_terms = [(term, score) for term, score in sorted_terms
                            if term not in stopwords]

        # Extract keywords
        keywords = [term for term, score in sorted_terms[:top_n]]

        return keywords

    except Exception as e:
        debug(config, f"Error extracting keywords with TF-IDF: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def extract_keywords_rake(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using RAKE-NLTK

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    if not require('rake_nltk', 'keyword extraction with RAKE-NLTK'):
        warning("rake_nltk not available. Install with: pip install rake-nltk")
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)

    debug(config, f"Extracting keywords with RAKE-NLTK from {len(texts)} texts in {language}")

    try:
        from rake_nltk import Rake
        import nltk

        # Process text based on language
        if language == 'zh':
            # Special handling for Chinese
            processed_text = preprocess_chinese_text(config, texts, segment=True, join=True)
        else:
            # For other languages
            processed_text = " ".join(texts)

        # Configure stopwords
        rake_stopwords = stopwords
        if rake_stopwords is None:
            # Download NLTK stopwords if necessary
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')

            if language == 'zh':
                # Use custom Chinese stopwords since NLTK doesn't have them
                rake_stopwords = load_chinese_stopwords()
            else:
                # Use NLTK's stopwords for other languages
                from nltk.corpus import stopwords as nltk_stopwords
                nltk_lang = 'english' if language == 'en' else language
                try:
                    rake_stopwords = set(nltk_stopwords.words(nltk_lang))
                except:
                    rake_stopwords = set(nltk_stopwords.words('english'))

        # Initialize Rake with appropriate stopwords
        rake = Rake(stopwords=rake_stopwords, include_repeated_phrases=False)

        # Extract keywords
        rake.extract_keywords_from_text(processed_text)

        # Get ranked phrases
        ranked_phrases = rake.get_ranked_phrases()[:top_n]

        # For Chinese, post-process the results to remove spaces if needed
        if language == 'zh':
            ranked_phrases = [phrase.replace(" ", "") for phrase in ranked_phrases]

        return ranked_phrases

    except Exception as e:
        debug(config, f"Error extracting keywords with RAKE-NLTK: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        # Fall back to TF-IDF
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)


def extract_keywords_yake(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using YAKE

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    if not require('yake', 'keyword extraction with YAKE'):
        warning("YAKE not available. Install with: pip install yake")
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)

    debug(config, f"Extracting keywords with YAKE from {len(texts)} texts in {language}")

    try:
        import yake

        # Process text based on language
        if language == 'zh':
            # Special handling for Chinese
            processed_text = preprocess_chinese_text(config, texts, segment=True, join=True)
        else:
            # For other languages
            processed_text = " ".join(texts)

        # Get max ngram size from config
        max_ngram_size = config.get('keywords.max_ngram_size', 2)

        # Configure language for YAKE
        language_map = {
            "en": "en",
            "zh": "zh",
            "fr": "fr",
            "pt": "pt",
            "it": "it",
            "nl": "nl",
            "de": "de",
            "es": "es"
            # Add other languages as needed
        }
        yake_language = language_map.get(language, "en")

        # Initialize keyword extractor
        kw_extractor = yake.KeywordExtractor(
            lan=yake_language,
            n=max_ngram_size,
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=1,
            top=top_n
        )

        # Extract keywords
        keywords = kw_extractor.extract_keywords(processed_text)

        # YAKE returns (keyword, score) where lower score is better
        # Sort by score (ascending) and return just the keywords
        sorted_keywords = sorted(keywords, key=lambda x: x[1])

        # For Chinese, post-process the results to remove spaces if needed
        if language == 'zh':
            return [kw[0].replace(" ", "") for kw in sorted_keywords[:top_n]]
        else:
            return [kw[0] for kw in sorted_keywords[:top_n]]

    except Exception as e:
        debug(config, f"Error extracting keywords with YAKE: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        # Fall back to TF-IDF
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)


def extract_keywords_keybert(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using KeyBERT

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    if not require('keybert', 'keyword extraction with KeyBERT'):
        warning("KeyBERT not available. Install with: pip install keybert")
        # Try falling back to YAKE, then to TF-IDF
        try:
            return extract_keywords_yake(config, texts, language, top_n, stopwords)
        except:
            return extract_keywords_tfidf(config, texts, language, top_n, stopwords)

    debug(config, f"Extracting keywords with KeyBERT from {len(texts)} texts in {language}")

    try:
        from keybert import KeyBERT

        # Process text based on language - for KeyBERT we don't join with spaces for Chinese
        if language == 'zh':
            processed_text = preprocess_chinese_text(config, texts, segment=False, join=True)
        else:
            processed_text = " ".join(texts)

        # Get max ngram size from config
        max_ngram_size = config.get('keywords.max_ngram_size', 2)

        # Select appropriate model based on language
        if language == "zh":
            # Models with good Chinese support
            model_options = [
                "paraphrase-multilingual-MiniLM-L12-v2",
                "distiluse-base-multilingual-cased-v2"
            ]
            model_name = model_options[0]
        else:
            # Default model for English and other languages
            model_name = "all-MiniLM-L6-v2"

        debug(config, f"Using model '{model_name}' for KeyBERT")

        # Initialize KeyBERT
        kw_model = KeyBERT(model=model_name)

        # Configure stopwords
        stop_words = None
        if language == 'en' and stopwords is None:
            stop_words = 'english'
        elif stopwords:
            stop_words = stopwords

        # For Chinese text, use custom tokenizer if available
        if language == "zh":
            extraction_result = _keybert_chinese_extraction(
                config, kw_model, processed_text, max_ngram_size, stop_words, top_n
            )
            if extraction_result:
                return extraction_result

        # Standard extraction for non-Chinese text or fallback for Chinese
        keywords = kw_model.extract_keywords(
            processed_text,
            keyphrase_ngram_range=(1, max_ngram_size),
            stop_words=stop_words,
            top_n=top_n,
            use_maxsum=True,
            diversity=0.5
        )

        # KeyBERT returns (keyword, score) tuples
        return [kw[0] for kw in keywords]

    except Exception as e:
        debug(config, f"Error extracting keywords with KeyBERT: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        # Fall back to TF-IDF
        return extract_keywords_tfidf(config, texts, language, top_n, stopwords)


def _keybert_chinese_extraction(config, kw_model, text, max_ngram_size, stop_words, top_n):
    """
    Helper function for Chinese-specific KeyBERT extraction

    Args:
        config: Configuration object
        kw_model: KeyBERT model instance
        text: Chinese text to process
        max_ngram_size: Maximum size of n-grams
        stop_words: Stopwords to filter
        top_n: Number of keywords to extract

    Returns:
        List[str]: Extracted keywords or None if unsuccessful
    """
    try:
        # Use spaCy for tokenization
        if SPACY_AVAILABLE:
            nlp = get_spacy_model('zh')
            if nlp:
                # Create a custom tokenizer function
                def chinese_tokenizer(text):
                    doc = nlp(text)
                    return [token.text for token in doc if not token.is_punct]

                # Extract keywords with custom tokenizer
                keywords = kw_model.extract_keywords(
                    text,
                    keyphrase_ngram_range=(1, max_ngram_size),
                    stop_words=stop_words,
                    top_n=top_n,
                    use_maxsum=True,
                    diversity=0.5,
                    # Pass custom tokenizer
                    vectorizer_kwargs={'tokenizer': chinese_tokenizer}
                )

                # Remove spaces from results
                return [kw[0].replace(" ", "") for kw in keywords]

        return None  # Return None to indicate fallback needed

    except Exception as e:
        debug(config, f"Error in Chinese-specific KeyBERT extraction: {str(e)}")
        return None


def preprocess_chinese_text(config, texts, segment=True, join=True):
    """
    Preprocess Chinese text for keyword extraction

    Args:
        config: Configuration object
        texts (List[str]): Input texts
        segment (bool): Whether to segment with spaCy
        join (bool): Whether to join words with spaces

    Returns:
        Union[str, List[str]]: Processed text(s)
    """
    try:
        # If no segmentation needed or spaCy not available
        if not segment or not SPACY_AVAILABLE:
            # Return as-is or joined
            return " ".join(texts) if join else texts

        nlp = get_spacy_model('zh')
        if not nlp:
            return " ".join(texts) if join else texts

        # Segment each text
        segmented_texts = []
        for text in texts:
            doc = nlp(text)
            if join:
                # Join with spaces to create format similar to English words
                segmented_text = " ".join([token.text for token in doc])
            else:
                # Keep as list of words
                segmented_text = [token.text for token in doc]
            segmented_texts.append(segmented_text)

        # Return segmented texts
        if join:
            return " ".join(segmented_texts) if len(segmented_texts) > 1 else segmented_texts[0]
        else:
            return segmented_texts

    except Exception as e:
        debug(config, f"Error preprocessing Chinese text: {str(e)}")
        return " ".join(texts) if join else texts


def load_chinese_stopwords():
    """
    Load Chinese stopwords from lexicon file

    Returns:
        Set[str]: Set of Chinese stopwords
    """
    stopwords = set()
    try:
        # Try to load from lexicon file
        with open(os.path.join('lexicon', 'stopwords_zh.txt'), 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
    except FileNotFoundError:
        # Basic fallback stopwords
        stopwords = {"的", "了", "和", "是", "就", "都", "而", "及", "与", "这", "那", "你", "我", "他"}

    return stopwords


def configure_vectorizer(config, doc_count, language=None, chinese_stopwords=None):
    """
    Create a TF-IDF vectorizer with parameters appropriate for corpus size and language

    Args:
        config: Configuration object for debug output
        doc_count (int): Number of documents in corpus
        language (str): Primary language of corpus
        chinese_stopwords (set): Set of Chinese stopwords

    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        error("scikit-learn not available. Install with: pip install scikit-learn")
        return None

    debug(config, f"Configuring TF-IDF vectorizer for {doc_count} documents in language: {language}")

    # Determine if Chinese corpus
    is_chinese = language == "zh"

    # Configure min_df and max_df based on corpus size
    if doc_count <= 2:
        # For tiny corpus
        min_df = 1
        max_df = 1.0
    elif doc_count <= 5:
        # For very small corpus
        min_df = 1
        max_df = 0.9
    elif doc_count <= 10:
        # For small corpus
        min_df = 1
        max_df = 0.8
    elif doc_count <= 50:
        # For medium corpus
        min_df = 2
        max_df = 0.8
    else:
        # For large corpus
        min_df = 2
        max_df = 0.7

    debug(config, f"Using min_df={min_df}, max_df={max_df}")

    # Configure language-specific parameters
    if is_chinese:
        try:
            # Create a tokenizer that uses spaCy
            def spacy_chinese_tokenizer(text):
                nlp = get_spacy_model('zh')
                if nlp:
                    doc = nlp(text)
                    return [token.text for token in doc if not token.is_punct]
                # Fallback to character-based
                return list(text)

            # Create vectorizer for Chinese text
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words=None,  # Use tokenizer-based filtering
                tokenizer=spacy_chinese_tokenizer
            )
        except Exception as e:
            error(f"Error creating Chinese vectorizer: {str(e)}")
            # Fallback to basic vectorizer
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df
            )
    else:
        # Create vectorizer for English text
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words="english"
        )

    return vectorizer