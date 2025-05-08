"""
Keyword extraction methods for document analysis toolkit
Uses SpaCy as primary extraction method with TF-IDF as fallback
"""

import os
import re
from typing import List, Set, Dict, Any, Optional, Callable
import numpy as np
from collections import Counter

from core.engine.logging import debug, warning, info, error, trace
from core.engine.dependencies import require, is_available
from core.language.spacy_tokenizer import SPACY_AVAILABLE, get_spacy_model, load_stopwords


def extract_noun_phrases(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract noun phrases using spaCy

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of phrases to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[Dict]: Extracted noun phrases with counts
    """
    debug(config, f"Extracting noun phrases from {len(texts)} texts in {language}")

    # Check spaCy availability
    if not SPACY_AVAILABLE:
        warning("spaCy not available for noun phrase extraction")
        return []

    if not texts:
        return []

    try:
        # Get appropriate spaCy model for language
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}")
            return []

        # Process each text and collect noun phrases
        phrase_counts = Counter()

        # Limit to a manageable number of texts
        for text in texts[:min(len(texts), 20)]:
            # Limit text length
            if len(text) > 10000:
                text = text[:10000]

            doc = nlp(text)

            # Extract noun chunks (noun phrases)
            for chunk in doc.noun_chunks:
                # Skip very short chunks
                if len(chunk.text) < 3:
                    continue

                # Skip chunks that are entirely stopwords
                if stopwords and all(token.text.lower() in stopwords for token in chunk):
                    continue

                # Clean up the phrase
                clean_words = [token.text for token in chunk
                              if not token.is_stop and not token.is_punct]
                if clean_words:
                    clean_phrase = " ".join(clean_words)
                    if len(clean_phrase) >= 3:  # At least 3 chars
                        phrase_counts[clean_phrase] += 1

        # Convert to list with details
        phrases = []
        for phrase, count in phrase_counts.most_common(top_n):
            phrases.append({
                'text': phrase,
                'count': count,
                'type': 'NOUN_PHRASE'
            })

        return phrases

    except Exception as e:
        debug(config, f"Error extracting noun phrases with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def extract_entities_from_text(config, texts, language="en", top_n=20, stopwords=None):
    """
    Extract named entities from text using spaCy

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of entities to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[Dict]: Extracted entities with type and count information
    """
    debug(config, f"Extracting entities from {len(texts)} texts in {language}")

    # Check spaCy availability
    if not SPACY_AVAILABLE:
        warning("spaCy not available for entity extraction")
        return []

    if not texts:
        return []

    try:
        # Get appropriate spaCy model for language
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}")
            return []

        # Combine texts with a length limit to avoid memory issues
        combined_text = " ".join(texts)
        if len(combined_text) > 100000:  # Limit text length to avoid memory issues
            combined_text = combined_text[:100000]

        # Process with spaCy
        doc = nlp(combined_text)

        # Extract named entities
        entity_data = []
        entity_counts = Counter()
        entity_types = {}

        # First pass - count entities
        for ent in doc.ents:
            # Skip very short entities and stopwords
            if len(ent.text) < 2 or (stopwords and ent.text.lower() in stopwords):
                continue

            entity_text = ent.text.strip()
            entity_counts[entity_text] += 1
            entity_types[entity_text] = ent.label_

        # Convert to list with details
        for entity_text, count in entity_counts.most_common(top_n):
            entity_data.append({
                'text': entity_text,
                'count': count,
                'type': entity_types.get(entity_text, "UNKNOWN")
            })

        # Add noun phrases if we have fewer than top_n entities
        if len(entity_data) < top_n:
            noun_phrases = extract_noun_phrases(config, texts, language, top_n - len(entity_data), stopwords)
            for phrase in noun_phrases:
                # Check if this phrase is already in entity data
                if not any(entity['text'] == phrase['text'] for entity in entity_data):
                    entity_data.append(phrase)

        return entity_data

    except Exception as e:
        debug(config, f"Error extracting entities with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def extract_key_phrases(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract key phrases using spaCy

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of phrases to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[Dict]: Extracted key phrases with counts
    """
    debug(config, f"Extracting key phrases from {len(texts)} texts in {language}")

    # Check spaCy availability
    if not SPACY_AVAILABLE:
        warning("spaCy not available for key phrase extraction")
        return []

    if not texts:
        return []

    try:
        # Get appropriate spaCy model for language
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}")
            return []

        # Combine texts with a length limit to avoid memory issues
        combined_text = " ".join(texts[:20])  # Limit to first 20 texts
        if len(combined_text) > 50000:  # Limit text length
            combined_text = combined_text[:50000]

        # Process with spaCy
        doc = nlp(combined_text)

        # Extract noun phrases and verb phrases
        phrase_counts = Counter()

        # Get noun chunks
        for chunk in doc.noun_chunks:
            # Skip stopwords and very short chunks
            if len(chunk.text) < 3:
                continue

            # Clean chunk text (remove stopwords)
            clean_words = [token.text for token in chunk if not token.is_stop and not token.is_punct]
            if clean_words:
                clean_phrase = " ".join(clean_words)
                if len(clean_phrase) >= 3:  # At least 3 chars
                    phrase_counts[clean_phrase] += 1

        # Get verb phrases (verb + direct object)
        for token in doc:
            if token.pos_ == "VERB":
                # Find direct objects of this verb
                for child in token.children:
                    if child.dep_ in ["dobj", "obj"]:
                        phrase = f"{token.lemma_} {child.text}"
                        phrase_counts[phrase] += 1

        # Convert to list with details
        phrases = []
        for phrase, count in phrase_counts.most_common(top_n):
            phrases.append({
                'text': phrase,
                'count': count,
                'type': 'PHRASE'
            })

        return phrases

    except Exception as e:
        debug(config, f"Error extracting key phrases with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def configure_vectorizer(config, doc_count, language=None, stopwords=None):
    """
    Create a TF-IDF vectorizer with parameters appropriate for corpus size and language

    Args:
        config: Configuration object for debug output
        doc_count (int): Number of documents in corpus
        language (str): Primary language of corpus
        stopwords (set): Set of stopwords

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

    # Configure language-specific parameters using spaCy
    if is_chinese and SPACY_AVAILABLE:
        try:
            # Create a tokenizer using spaCy for Chinese
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
            return vectorizer
        except Exception as e:
            error(f"Error creating Chinese vectorizer: {str(e)}")
            # Fall back to basic vectorizer

    # For non-Chinese languages or if spaCy failed
    if language == 'en' or not is_chinese:
        # Create vectorizer for English text
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=stopwords if stopwords else "english"
        )
        return vectorizer
    else:
        # Basic vectorizer for any language
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df
        )
        return vectorizer


def extract_keywords_tfidf(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using TF-IDF (fallback method)

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


def extract_keywords(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords using SpaCy as primary approach

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

    # Primary method is always spaCy
    if method == 'spacy' or method == 'default':
        return extract_keywords_spacy(config, texts, language, top_n, stopwords)
    # Fallback to TF-IDF which doesn't require additional libraries
    else:
        # Warn about deprecated methods
        warning(f"Method '{method}' is deprecated. Using SpaCy-based extraction instead.")
        return extract_keywords_spacy(config, texts, language, top_n, stopwords)


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
    Extract keywords using TF-IDF (fallback method)

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


def extract_entities_from_text(config, texts, language="en", top_n=20, stopwords=None):
    """
    Extract named entities from text using spaCy

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of entities to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[Dict]: Extracted entities with type and count information
    """
    debug(config, f"Extracting entities from {len(texts)} texts in {language}")

    # Check spaCy availability
    if not SPACY_AVAILABLE:
        warning("spaCy not available for entity extraction")
        return []

    if not texts:
        return []

    try:
        # Get appropriate spaCy model for language
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}")
            return []

        # Combine texts with a length limit to avoid memory issues
        combined_text = " ".join(texts)
        if len(combined_text) > 100000:  # Limit text length to avoid memory issues
            combined_text = combined_text[:100000]

        # Process with spaCy
        doc = nlp(combined_text)

        # Extract named entities
        entity_data = []
        entity_counts = Counter()
        entity_types = {}

        # First pass - count entities
        for ent in doc.ents:
            # Skip very short entities and stopwords
            if len(ent.text) < 2 or (stopwords and ent.text.lower() in stopwords):
                continue

            entity_text = ent.text.strip()
            entity_counts[entity_text] += 1
            entity_types[entity_text] = ent.label_

        # Convert to list with details
        for entity_text, count in entity_counts.most_common(top_n):
            entity_data.append({
                'text': entity_text,
                'count': count,
                'type': entity_types.get(entity_text, "UNKNOWN")
            })

        # Add noun phrases if we have fewer than top_n entities
        if len(entity_data) < top_n:
            noun_phrases = extract_noun_phrases(config, texts, language, top_n - len(entity_data), stopwords)
            for phrase in noun_phrases:
                # Check if this phrase is already in entity data
                if not any(entity['text'] == phrase['text'] for entity in entity_data):
                    entity_data.append(phrase)

        return entity_data

    except Exception as e:
        debug(config, f"Error extracting entities with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def extract_key_phrases(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract key phrases using spaCy

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of phrases to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[Dict]: Extracted key phrases with counts
    """
    debug(config, f"Extracting key phrases from {len(texts)} texts in {language}")

    # Check spaCy availability
    if not SPACY_AVAILABLE:
        warning("spaCy not available for key phrase extraction")
        return []

    if not texts:
        return []

    try:
        # Get appropriate spaCy model for language
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}")
            return []

        # Combine texts with a length limit to avoid memory issues
        combined_text = " ".join(texts[:20])  # Limit to first 20 texts
        if len(combined_text) > 50000:  # Limit text length
            combined_text = combined_text[:50000]

        # Process with spaCy
        doc = nlp(combined_text)

        # Extract noun phrases and verb phrases
        phrase_counts = Counter()

        # Get noun chunks
        for chunk in doc.noun_chunks:
            # Skip stopwords and very short chunks
            if len(chunk.text) < 3:
                continue

            # Clean chunk text (remove stopwords)
            clean_words = [token.text for token in chunk if not token.is_stop and not token.is_punct]
            if clean_words:
                clean_phrase = " ".join(clean_words)
                if len(clean_phrase) >= 3:  # At least 3 chars
                    phrase_counts[clean_phrase] += 1

        # Get verb phrases (verb + direct object)
        for token in doc:
            if token.pos_ == "VERB":
                # Find direct objects of this verb
                for child in token.children:
                    if child.dep_ in ["dobj", "obj"]:
                        phrase = f"{token.lemma_} {child.text}"
                        phrase_counts[phrase] += 1

        # Convert to list with details
        phrases = []
        for phrase, count in phrase_counts.most_common(top_n):
            phrases.append({
                'text': phrase,
                'count': count,
                'type': 'PHRASE'
            })

        return phrases

    except Exception as e:
        debug(config, f"Error extracting key phrases with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def extract_noun_phrases(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract noun phrases using spaCy

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language code
        top_n (int): Number of phrases to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[Dict]: Extracted noun phrases with counts
    """
    debug(config, f"Extracting noun phrases from {len(texts)} texts in {language}")

    # Check spaCy availability
    if not SPACY_AVAILABLE:
        warning("spaCy not available for noun phrase extraction")
        return []

    if not texts:
        return []

    try:
        # Get appropriate spaCy model for language
        nlp = get_spacy_model(language)
        if not nlp:
            warning(f"No spaCy model available for {language}")
            return []

        # Process each text and collect noun phrases
        phrase_counts = Counter()

        # Limit to a manageable number of texts
        for text in texts[:min(len(texts), 20)]:
            # Limit text length
            if len(text) > 10000:
                text = text[:10000]

            doc = nlp(text)

            # Extract noun chunks (noun phrases)
            for chunk in doc.noun_chunks:
                # Skip very short chunks
                if len(chunk.text) < 3:
                    continue

                # Skip chunks that are entirely stopwords
                if stopwords and all(token.text.lower() in stopwords for token in chunk):
                    continue

                # Clean up the phrase
                clean_words = [token.text for token in chunk
                              if not token.is_stop and not token.is_punct]
                if clean_words:
                    clean_phrase = " ".join(clean_words)
                    if len(clean_phrase) >= 3:  # At least 3 chars
                        phrase_counts[clean_phrase] += 1

        # Convert to list with details
        phrases = []
        for phrase, count in phrase_counts.most_common(top_n):
            phrases.append({
                'text': phrase,
                'count': count,
                'type': 'NOUN_PHRASE'
            })

        return phrases

    except Exception as e:
        debug(config, f"Error extracting noun phrases with spaCy: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        return []


def configure_vectorizer(config, doc_count, language=None, stopwords=None):
    """
    Create a TF-IDF vectorizer with parameters appropriate for corpus size and language

    Args:
        config: Configuration object for debug output
        doc_count (int): Number of documents in corpus
        language (str): Primary language of corpus
        stopwords (set): Set of stopwords

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

    # Configure language-specific parameters using spaCy
    if is_chinese and SPACY_AVAILABLE:
        try:
            # Create a tokenizer using spaCy for Chinese
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
            return vectorizer
        except Exception as e:
            error(f"Error creating Chinese vectorizer: {str(e)}")
            # Fall back to basic vectorizer

    # For non-Chinese languages or if spaCy failed
    if language == 'en' or not is_chinese:
        # Create vectorizer for English text
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words=stopwords if stopwords else "english"
        )
        return vectorizer
    else:
        # Basic vectorizer for any language
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df
        )
        return vectorizer