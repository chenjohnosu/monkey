"""
Theme analysis module with enhanced content-based processing
"""

import os
import sys
import io
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans

from core.engine.utils import (
    ensure_dir, get_file_content, timestamp_filename, format_size,
    split_text_into_chunks, configure_vectorizer, extract_keywords
)
from core.engine.logging import debug_print, warning, info, error
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.language.tokenizer import ChineseTokenizer, JIEBA_AVAILABLE
from core.engine.common import safe_execute

# Conditional imports with improved handling
JIEBA_AVAILABLE = False
jieba = None

try:
    original_stdout = sys.stdout
    sys.stdout = io.StringIO()  # Redirect to string buffer

    import jieba
    jieba.initialize()  # Explicitly initialize once

    # Restore stdout
    sys.stdout = original_stdout

    JIEBA_AVAILABLE = True
except ImportError:
    warning("jieba not available. Chinese text processing will be limited.")
    jieba = None

try:
    from community import best_partition
except ImportError:
    best_partition = None

class ThemeAnalyzer:
    """Analyzes themes in document content with enhanced processing capabilities"""

    # Add to ThemeAnalyzer's __init__ method in core/modes/themes.py

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the theme analyzer with shared components"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)

        # Initialize LLM connector factory
        try:
            from core.connectors.connector_factory import ConnectorFactory
            self.factory = ConnectorFactory(config)
            debug_print(config, "LLM connector factory initialized for theme analysis")
        except Exception as e:
            debug_print(config, f"Error initializing ConnectorFactory: {str(e)}")
            self.factory = None

        # Prepare Chinese and English stopwords
        self._prepare_stopwords()

        if JIEBA_AVAILABLE:
            try:
                # Redirect jieba's stdout to suppress initialization messages
                import sys
                import io
                original_stdout = sys.stdout
                sys.stdout = io.StringIO()  # Redirect to string buffer

                # Pre-initialize jieba
                import jieba
                jieba.initialize()

                # Restore stdout
                sys.stdout = original_stdout

                debug_print(config, "Jieba initialized for Chinese text processing")
            except Exception as e:
                debug_print(config, f"Error initializing jieba: {str(e)}")

        debug_print(config, "Theme analyzer initialized")

    def _prepare_stopwords(self):
        """Prepare stopwords for processing"""
        self.chinese_stopwords = set()
        self.english_stopwords = set()

        # Load Chinese stopwords
        try:
            with open(os.path.join('lexicon', 'stopwords_zh.txt'), 'r', encoding='utf-8') as f:
                self.chinese_stopwords = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            self.chinese_stopwords = {"的", "了", "和", "是", "就", "都", "而", "及"}

        # Load English stopwords
        try:
            with open(os.path.join('lexicon', 'stopwords_en.txt'), 'r', encoding='utf-8') as f:
                self.english_stopwords = set(line.strip() for line in f if line.strip())
        except FileNotFoundError:
            self.english_stopwords = {
                "the", "and", "a", "to", "of", "in", "is", "that", "it",
                "with", "for", "as", "was", "on", "are", "by", "this"
            }

    def analyze(self, workspace: str, method: str = 'all') -> Dict[str, Any]:
        """
        Analyze themes in a workspace using multiple methods

        Args:
            workspace (str): The workspace to analyze
            method (str): Analysis method ('all', 'nfm', 'net', 'key', 'lsa', 'cluster')

        Returns:
            Dict: Analyzed themes
        """
        debug_print(self.config, f"Analyzing themes in workspace '{workspace}' using method '{method}'")

        # Validate method
        valid_methods = ['all', 'nfm', 'net', 'key', 'lsa', 'cluster']
        if method not in valid_methods:
            print(f"Invalid method: {method}. Must be one of: {', '.join(valid_methods)}")
            return {}

        # Load documents
        docs = self.storage_manager.get_documents(workspace)
        if not docs:
            print(f"No documents found in workspace '{workspace}'")
            return {}

        # Extract and preprocess document content
        doc_contents, doc_languages = self._extract_document_contents(docs)

        # Display language distribution
        self._display_language_stats(doc_languages)

        # Run selected analysis methods
        results = {}

        method_map = {
            'nfm': self._analyze_named_entities,
            'net': self._analyze_content_network,
            'key': self._analyze_content_keywords,
            'lsa': self._analyze_latent_semantics,
            'cluster': self._analyze_document_clusters
        }

        for analysis_method, handler in method_map.items():
            if method in ['all', analysis_method]:
                try:
                    results[analysis_method] = handler(doc_contents)
                except Exception as e:
                    error(f"Error in {analysis_method} analysis: {str(e)}")
                    import traceback
                    debug_print(self.config, traceback.format_exc())

        # Output results
        self._output_results(workspace, results, method)

        return results

    def _extract_document_contents(self, docs: List[Dict]) -> Tuple[List[Dict], Counter]:
        """
        Extract and preprocess document contents

        Args:
            docs (List[Dict]): Raw documents

        Returns:
            Tuple of processed documents and language counts
        """
        doc_contents = []
        doc_languages = Counter()

        for doc in docs:
            # Extract content and metadata
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            language = doc.get("metadata", {}).get("language", "en")

            # Preprocess content
            try:
                processed = self.text_processor.preprocess(content)

                # Skip empty documents
                if not processed['processed']:
                    continue

                # Track language
                doc_languages[language] += 1

                doc_contents.append({
                    "id": len(doc_contents),
                    "source": source,
                    "language": language,
                    "content": content,
                    "processed_content": processed['processed']
                })
            except Exception as e:
                debug_print(self.config, f"Error preprocessing document: {str(e)}")

        return doc_contents, doc_languages

    def _display_language_stats(self, language_counts: Counter):
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

    def _output_results(self, workspace: str, results: Dict, method: str):
        """
        Output theme analysis results using output manager with improved keyword handling

        Args:
            workspace (str): Workspace name
            results (Dict): Analysis results
            method (str): Analysis method
        """
        # Use output_manager for formatted display
        self.output_manager.print_formatted('header', "THEME ANALYSIS RESULTS")

        # Display results for each method
        for m, result in results.items():
            self.output_manager.print_formatted('subheader', result.get('method', m))

            # Display method-specific statistics
            if 'entity_count' in result:
                self.output_manager.print_formatted('kv', result['entity_count'], key="Total entities")
            if 'variance_explained' in result:
                self.output_manager.print_formatted('kv', f"{result['variance_explained']}%", key="Variance explained")
            if 'clusters' in result:
                self.output_manager.print_formatted('kv', result['clusters'], key="Number of clusters")
            if 'error' in result:
                self.output_manager.print_formatted('feedback', f"Error: {result['error']}", success=False)

            # Display themes
            themes = result.get('themes', [])
            if not themes:
                self.output_manager.print_formatted('feedback', "No themes identified", success=False)
                continue

            print(f"\nFound {len(themes)} themes/keywords")

            for theme in themes:
                # Different handling based on theme type
                if 'keyword' in theme:
                    # This is a keyword theme from content keyword analysis
                    self.output_manager.print_formatted('mini_header', f"Keyword: {theme['keyword']}")

                    if 'score' in theme:
                        self.output_manager.print_formatted('kv', f"{theme['score']:.4f}", key="Score")

                    if 'documents' in theme:
                        self.output_manager.print_formatted('kv', theme['documents'], key="Documents")

                    # Show documents list if available
                    if 'documents_list' in theme and isinstance(theme['documents_list'], list):
                        print("\n  Document sources:")
                        for doc in theme['documents_list'][:5]:
                            self.output_manager.print_formatted('list', str(doc), indent=4)
                        if len(theme['documents_list']) > 5:
                            print(f"  ... and {len(theme['documents_list']) - 5} more")

                elif 'name' in theme:
                    # This is a standard theme
                    self.output_manager.print_formatted('mini_header', theme['name'])

                    # Keywords
                    if 'keywords' in theme:
                        self.output_manager.print_formatted('kv', ', '.join(theme['keywords']), key="Keywords")

                    # Various metrics
                    metrics = [
                        ('score', 'Score'),
                        ('frequency', 'Frequency'),
                        ('centrality', 'Centrality'),
                        ('document_count', 'Documents')
                    ]
                    for key, label in metrics:
                        if key in theme:
                            self.output_manager.print_formatted('kv', theme[key], key=label)

                    # Documents
                    if 'documents' in theme and isinstance(theme['documents'], list):
                        print("\n  Document files:")
                        for doc in theme['documents'][:5]:
                            self.output_manager.print_formatted('list', str(doc), indent=4)
                        if len(theme['documents']) > 5:
                            print(f"  ... and {len(theme['documents']) - 5} more")

                    # Descriptions
                    if 'description' in theme and theme['description']:
                        print("\n  Description:")
                        print(f"  {theme['description']}")

        # Save results to file
        output_format = self.config.get('system.output_format', 'txt')
        filepath = self.output_manager.save_theme_analysis(workspace, results, method, output_format)

        # Show success message
        self.output_manager.print_formatted('feedback', f"Results saved to: {filepath}")

    def _extract_english_keywords(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract keywords from English document content with improved error handling

        Args:
            docs (List[Dict]): English documents

        Returns:
            List[Dict]: Extracted keywords
        """
        # Extract document texts
        doc_texts = [doc["processed_content"] for doc in docs]
        doc_sources = [doc["source"] for doc in docs]

        # Log extraction attempt
        print(f"Extracting keywords from {len(doc_texts)} English documents")

        # Create mapping to store document sources for each keyword
        keyword_doc_mapping = defaultdict(list)

        # Check if documents contain content
        if not doc_texts or all(not text for text in doc_texts):
            print("No valid content found in English documents")
            return []

        try:
            # Use TF-IDF vectorizer to identify important terms
            vectorizer = TfidfVectorizer(
                min_df=1,  # Lower min_df for small document sets
                max_df=0.95,
                stop_words="english"
            )

            # Fit TF-IDF on documents
            tfidf_matrix = vectorizer.fit_transform(doc_texts)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            print(f"Extracted {len(feature_names)} unique terms from English documents")

            # Calculate average TF-IDF scores across documents
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            # Count documents containing each term
            term_doc_counts = defaultdict(int)
            for doc_id, doc_text in enumerate(doc_texts):
                # Create a set of terms in this document
                doc_terms = set(doc_text.split())

                # Check each term in feature names
                for term in feature_names:
                    if term in doc_terms:
                        term_doc_counts[term] += 1
                        keyword_doc_mapping[term].append(doc_sources[doc_id])

            # Create keyword list
            keywords = []
            for i, term in enumerate(feature_names):
                # Skip very short terms
                if len(term) < 3:
                    continue

                score = avg_scores[i]
                doc_count = term_doc_counts.get(term, 0)

                # Only include terms that appear in at least one document
                if doc_count > 0:
                    keywords.append({
                        "keyword": term,
                        "score": float(score),  # Convert numpy float to Python float
                        "documents": doc_count,
                        "doc_sources": keyword_doc_mapping.get(term, [])
                    })

            # Sort by score and document count
            keywords.sort(key=lambda x: (x["score"], x["documents"]), reverse=True)

            print(f"Found {len(keywords)} keywords in English documents")

            # Limit to top keywords
            return keywords[:30]  # Return more keywords to ensure we have enough after filtering

        except Exception as e:
            print(f"Error extracting English keywords: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def _extract_chinese_keywords(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract keywords from Chinese document content with improved error handling

        Args:
            docs (List[Dict]): Chinese documents

        Returns:
            List[Dict]: Extracted keywords
        """
        # Log extraction attempt
        print(f"Extracting keywords from {len(docs)} Chinese documents")

        # Check if documents contain content
        if not docs or all(not doc.get("processed_content") for doc in docs):
            print("No valid content found in Chinese documents")
            return []

        try:
            # Count frequencies of character and word n-grams
            character_counts = Counter()
            bigram_counts = Counter()
            trigram_counts = Counter()

            # Document sources for tracking
            doc_sources = [doc["source"] for doc in docs]

            # Count frequencies
            for doc in docs:
                text = doc["processed_content"]
                if not text:
                    continue

                # Count characters
                character_counts.update(text)

                # Count n-grams
                for i in range(len(text) - 1):
                    # Bigrams
                    if i < len(text) - 1:
                        bigram = text[i:i + 2]
                        bigram_counts[bigram] += 1

                    # Trigrams
                    if i < len(text) - 2:
                        trigram = text[i:i + 3]
                        trigram_counts[trigram] += 1

            # Track document sources for each n-gram
            bigram_doc_sources = defaultdict(list)
            trigram_doc_sources = defaultdict(list)

            for doc_idx, doc in enumerate(docs):
                text = doc["processed_content"]
                if not text:
                    continue

                # Track bigrams
                for i in range(len(text) - 1):
                    if i < len(text) - 1:
                        bigram = text[i:i + 2]
                        bigram_doc_sources[bigram].append(doc_sources[doc_idx])

                # Track trigrams
                for i in range(len(text) - 2):
                    if i < len(text) - 2:
                        trigram = text[i:i + 3]
                        trigram_doc_sources[trigram].append(doc_sources[doc_idx])

            # Combine and sort keywords
            keywords = []

            # Add top bigrams
            for bigram, count in bigram_counts.most_common(15):
                if len(bigram) < 2 or count < 2:
                    continue

                # Check if bigram is in stopwords
                if bigram in self.chinese_stopwords:
                    continue

                score = count / sum(bigram_counts.values()) if bigram_counts else 0
                doc_sources_list = list(set(bigram_doc_sources[bigram]))

                keywords.append({
                    "keyword": bigram,
                    "score": float(score),  # Ensure we use a regular Python float
                    "documents": len(doc_sources_list),
                    "doc_sources": doc_sources_list
                })

            # Add top trigrams
            for trigram, count in trigram_counts.most_common(15):
                if len(trigram) < 3 or count < 2:
                    continue

                # Skip trigrams that are entirely in stopwords
                if all(char in self.chinese_stopwords for char in trigram):
                    continue

                score = count / sum(trigram_counts.values()) if trigram_counts else 0
                doc_sources_list = list(set(trigram_doc_sources[trigram]))

                keywords.append({
                    "keyword": trigram,
                    "score": float(score),  # Ensure we use a regular Python float
                    "documents": len(doc_sources_list),
                    "doc_sources": doc_sources_list
                })

            print(f"Found {len(keywords)} keywords in Chinese documents (bigrams and trigrams)")

            # If jieba is available, try to extract word-based keywords
            if JIEBA_AVAILABLE and jieba and len(docs) >= 2:
                print("Using jieba for additional Chinese keyword extraction")

                # Collect all text
                all_text = " ".join([doc["processed_content"] for doc in docs if doc.get("processed_content")])

                # Tokenize with jieba - using the already initialized instance
                words = list(jieba.cut(all_text))
                word_counts = Counter(words)

                # Add top words
                for word, count in word_counts.most_common(15):
                    # Skip short words and stopwords
                    if len(word) < 2 or word in self.chinese_stopwords:
                        continue

                    # Skip non-Chinese words
                    if not any('\u4e00' <= char <= '\u9fff' for char in word):
                        continue

                    score = count / len(words) if words else 0

                    # Determine which documents contain this word
                    doc_sources_list = []
                    for doc_idx, doc in enumerate(docs):
                        if doc.get("processed_content") and word in doc["processed_content"]:
                            doc_sources_list.append(doc_sources[doc_idx])

                    keywords.append({
                        "keyword": word,
                        "score": float(score),
                        "documents": len(doc_sources_list),
                        "doc_sources": doc_sources_list
                    })

            # Sort keywords by score
            keywords.sort(key=lambda x: x["score"], reverse=True)

            # Return top keywords
            return keywords[:30]  # Return more keywords to ensure we have enough after filtering

        except Exception as e:
            print(f"Error extracting Chinese keywords: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def _analyze_content_keywords(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Extract and analyze keywords from document content

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug_print(self.config, "Analyzing content keywords")

        # Group documents by language
        docs_by_language = defaultdict(list)
        for doc in doc_contents:
            language = doc["language"]
            docs_by_language[language].append(doc)

        # Extract keywords by language
        all_keywords = []

        # Print information about document grouping
        print(
            f"Documents grouped by language: {', '.join([f'{lang}: {len(docs)}' for lang, docs in docs_by_language.items()])}")

        # Process each language group
        for language, docs in docs_by_language.items():
            if language == "zh":
                print(f"Processing {len(docs)} Chinese documents")
                keywords = self._extract_chinese_keywords(docs)
            else:
                print(f"Processing {len(docs)} documents in language: {language}")
                keywords = self._extract_english_keywords(docs)

            print(f"Extracted {len(keywords)} keywords for language: {language}")
            all_keywords.extend(keywords)

        # Sort keywords by score
        all_keywords.sort(key=lambda x: x["score"], reverse=True)

        # Take top keywords
        top_keywords = all_keywords[:20]

        print(f"Final keyword count: {len(top_keywords)}")

        # Create proper themes format with keyword info
        formatted_keywords = []
        for kw in top_keywords:
            formatted_kw = {
                "keyword": kw["keyword"],
                "score": kw["score"],
                "documents": kw["documents"]
            }
            # Include document sources if available
            if "doc_sources" in kw:
                formatted_kw["documents_list"] = kw["doc_sources"]

            formatted_keywords.append(formatted_kw)

        return {
            "method": "Content Keyword Analysis",
            "themes": formatted_keywords
        }

    def _analyze_content_network(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Analyze content relationships between documents

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug_print(self.config, "Analyzing content relationships between documents")

        # Check if we have enough documents
        if len(doc_contents) < 2:
            return {
                "method": "Content Network Analysis",
                "themes": [{"name": "Insufficient documents for analysis", "centrality": 0, "nodes": []}]
            }

        # Extract document texts and metadata
        doc_texts = [doc["processed_content"] for doc in doc_contents]
        doc_metadata = [
            {
                "id": doc["id"],
                "source": doc["source"],
                "language": doc["language"]
            } for doc in doc_contents
        ]

        # Determine primary language
        languages = [meta["language"] for meta in doc_metadata]
        primary_language = Counter(languages).most_common(1)[0][0]
        is_chinese = primary_language == "zh"

        try:
            # Configure vectorizer based on language
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.chinese_stopwords if is_chinese else None
            )

            # Generate document vectors
            tfidf_matrix = vectorizer.fit_transform(doc_texts)

            # Calculate document similarity
            from sklearn.metrics.pairwise import cosine_similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Build document similarity network
            doc_network = nx.Graph()

            # Add nodes for documents
            for i, meta in enumerate(doc_metadata):
                doc_network.add_node(i, **meta)

            # Add edges for similar documents
            min_similarity = 0.1  # Lower threshold for small document sets

            for i in range(len(doc_metadata)):
                for j in range(i + 1, len(doc_metadata)):
                    similarity = similarity_matrix[i, j]

                    if similarity > min_similarity:
                        doc_network.add_edge(i, j, weight=similarity)

            # Remove isolated nodes
            doc_network.remove_nodes_from(list(nx.isolates(doc_network)))

            # Check if we have enough connected documents
            if doc_network.number_of_nodes() < 2:
                return {
                    "method": "Content Network Analysis",
                    "themes": [{"name": "Insufficient connected documents", "centrality": 0, "nodes": []}]
                }

            # Detect document communities
            try:
                # Try Louvain method first
                partition = best_partition(doc_network)
            except Exception:
                # Fallback to connected components
                partition = {}
                for i, component in enumerate(nx.connected_components(doc_network)):
                    for node in component:
                        partition[node] = i

            # Group documents by community
            communities = defaultdict(list)
            for doc_id, community_id in partition.items():
                communities[community_id].append(doc_id)

            # Extract themes from each community
            themes = []

            for community_id, doc_ids in communities.items():
                if len(doc_ids) < 1:
                    continue

                # Extract theme keywords from community documents
                community_texts = [doc_texts[doc_id] for doc_id in doc_ids]
                keywords = extract_keywords(
                    self.config,
                    community_texts,
                    language=primary_language,
                    top_n=5,
                    stopwords=self.chinese_stopwords if is_chinese else self.english_stopwords
                )

                # Get document sources
                doc_sources = [doc_metadata[doc_id]["source"] for doc_id in doc_ids]

                # Calculate community centrality
                community_subgraph = doc_network.subgraph(doc_ids)
                try:
                    centrality = nx.eigenvector_centrality(community_subgraph)
                    avg_centrality = sum(centrality.values()) / len(centrality)
                except:
                    avg_centrality = 0.5  # Default if calculation fails

                # Create theme
                theme_name = f"Theme: {', '.join(keywords[:3])}"

                themes.append({
                    "name": theme_name,
                    "centrality": round(avg_centrality, 2),
                    "nodes": doc_sources,
                    "keywords": keywords,
                    "size": len(doc_ids)
                })

            # Sort themes by size and centrality
            themes.sort(key=lambda x: (x["size"], x["centrality"]), reverse=True)

            return {
                "method": "Content Network Analysis",
                "themes": themes
            }

        except Exception as e:
            debug_print(self.config, f"Error in content network analysis: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())

            return {
                "method": "Content Network Analysis",
                "error": str(e),
                "themes": [{"name": "Analysis Error", "centrality": 0, "nodes": []}]
            }

    def _analyze_named_entities(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Extract and analyze named entities from document content

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Named entity analysis results
        """
        debug_print(self.config, "Analyzing named entities")

        # Detailed logging of input documents
        print(f"Total input documents: {len(doc_contents)}")
        for doc in doc_contents[:5]:  # Log first 5 documents for inspection
            print(f"Document source: {doc.get('source', 'Unknown')}")
            print(f"Document language: {doc.get('language', 'Unknown')}")
            print(f"Content length: {len(doc.get('processed_content', ''))}")
            print(f"First 100 chars: {doc.get('processed_content', '')[:100]}")

        # Group documents by language
        docs_by_language = defaultdict(list)
        for doc in doc_contents:
            language = doc.get("language", "en")
            docs_by_language[language].append(doc)

        # Prepare results
        all_entities = []
        total_entity_counts = defaultdict(int)

        # Process entities for each language
        for language, docs in docs_by_language.items():
            print(f"\nProcessing {language} language documents. Total: {len(docs)}")

            try:
                # Extract entities based on language
                if language == "zh":
                    language_entities = self._extract_chinese_entities(docs)
                else:
                    language_entities = self._extract_english_entities(docs)

                print(f"Extracted {len(language_entities)} entities for {language}")

                # Accumulate entities
                all_entities.extend(language_entities)

                # Count entities
                for entity in language_entities:
                    total_entity_counts[entity['value']] += entity['count']
            except Exception as e:
                print(f"Error extracting entities for {language}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Sort entities by overall frequency
        sorted_entities = sorted(
            [
                {
                    'value': entity,
                    'count': count,
                    'documents': sum(1 for doc in doc_contents if entity in doc.get('processed_content', ''))
                }
                for entity, count in total_entity_counts.items()
            ],
            key=lambda x: x['count'],
            reverse=True
        )

        print(f"\nTotal unique entities found: {len(sorted_entities)}")
        print("Top 10 entities:")
        for entity in sorted_entities[:10]:
            print(f"  {entity['value']}: count={entity['count']}, documents={entity['documents']}")

        # Generate themes from top entities
        themes = []
        for top_entity in sorted_entities[:20]:  # Top 20 entities
            themes.append({
                'name': f"Theme: {top_entity['value']}",
                'keywords': [top_entity['value']],
                'frequency': top_entity['count'],
                'document_count': top_entity['documents']
            })

        return {
            "method": "Named Entity Analysis",
            "entity_count": len(sorted_entities),
            "significant_entities": len(sorted_entities),
            "themes": themes
        }

    def _extract_chinese_entities(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract named entities from Chinese text with improved detection

        Args:
            docs (List[Dict]): Chinese documents

        Returns:
            List[Dict]: Extracted entities
        """
        entities = []

        # Use more comprehensive entity detection
        entity_patterns = [
            # Names (2-4 characters)
            r'[\u4e00-\u9fff]{2,4}[先生|女士|醫師|教授|博士|護士|老師|醫生]',
            # Organizations and institutions
            r'[\u4e00-\u9fff]{2,6}(醫院|診所|學校|協會|公司|機構|中心)',
            # Specialized terms
            r'[\u4e00-\u9fff]{2,6}(系統|方案|計畫|研究|專案|服務|領域)',
            # Location names
            r'[\u4e00-\u9fff]{2,6}(市|縣|省|區|鄉|鎮|路|街|醫學中心)'
        ]

        # Compile patterns
        import re
        compiled_patterns = [re.compile(pattern) for pattern in entity_patterns]

        # Process each document
        for doc in docs:
            text = doc.get('processed_content', '')

            # Use jieba for word segmentation if available
            # Note: jieba should already be initialized at this point
            if JIEBA_AVAILABLE and jieba:
                # Use the global jieba instance that was already initialized
                words = list(jieba.cut(text))
            else:
                # Fallback to character-based extraction
                words = list(text)

            # Collect entities
            doc_entities = []
            for pattern in compiled_patterns:
                doc_entities.extend(pattern.findall(text))

            # Additional character-based filtering
            char_entities = [
                word for word in words
                if (2 <= len(word) <= 4 and
                    all('\u4e00-\u9fff' in char for char in word) and
                    word not in self.chinese_stopwords)
            ]
            doc_entities.extend(char_entities)

            # Count entity frequencies
            entity_counts = Counter(doc_entities)

            # Convert to standardized format
            doc_entity_list = [
                {
                    'value': entity,
                    'count': count,
                    'documents': 1
                }
                for entity, count in entity_counts.items()
                if count > 1  # Only keep entities appearing more than once
            ]

            entities.extend(doc_entity_list)

        # Remove duplicates while preserving count information
        unique_entities = {}
        for entity in entities:
            if entity['value'] not in unique_entities:
                unique_entities[entity['value']] = entity
            else:
                unique_entities[entity['value']]['count'] += entity['count']
                unique_entities[entity['value']]['documents'] += entity['documents']

        return list(unique_entities.values())

    def _extract_english_entities(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract named entities from English text with improved detection

        Args:
            docs (List[Dict]): English documents

        Returns:
            List[Dict]: Extracted entities
        """
        import re

        entities = []

        # More comprehensive entity patterns
        entity_patterns = [
            # Proper names (potential person names)
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b',
            # Organizations
            r'\b([A-Z][a-z]+\s+(?:University|College|Institute|Hospital|Company|Corporation|Center))\b',
            # Technical terms and acronyms
            r'\b([A-Z]{2,})\b',
            # Specialized domain terms
            r'\b([A-Z][a-z]+(?:ing|tion|ment|ity|ship))\b'
        ]

        # Compile patterns
        compiled_patterns = [re.compile(pattern) for pattern in entity_patterns]

        for doc in docs:
            text = doc.get('processed_content', '')

            # Collect entities from different patterns
            doc_entities = []
            for pattern in compiled_patterns:
                doc_entities.extend(pattern.findall(text))

            # Additional term extraction
            technical_terms = re.findall(
                r'\b[A-Za-z]+(?:-[A-Za-z]+)+\b',  # Hyphenated terms
                text
            )
            camel_case_terms = re.findall(
                r'\b[a-z]+[A-Z][a-zA-Z]+\b',  # camelCase terms
                text
            )

            # Combine and filter entities
            doc_entities.extend(technical_terms)
            doc_entities.extend(camel_case_terms)

            # Filter out stopwords and short/irrelevant terms
            filtered_entities = [
                entity for entity in doc_entities
                if (len(entity) > 3 and
                    entity.lower() not in self.english_stopwords and
                    not entity.isdigit())
            ]

            # Count entity frequencies
            entity_counts = Counter(filtered_entities)

            # Convert to standardized format
            doc_entity_list = [
                {
                    'value': entity,
                    'count': count,
                    'documents': 1
                }
                for entity, count in entity_counts.items()
                if count > 1  # Only keep entities appearing more than once
            ]

            entities.extend(doc_entity_list)

        # Remove duplicates while preserving count information
        unique_entities = {}
        for entity in entities:
            if entity['value'] not in unique_entities:
                unique_entities[entity['value']] = entity
            else:
                unique_entities[entity['value']]['count'] += entity['count']
                unique_entities[entity['value']]['documents'] += entity['documents']

        return list(unique_entities.values())

    def _analyze_latent_semantics(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Analyze latent semantic themes using LSA/SVD

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Latent semantic analysis results
        """
        debug_print(self.config, "Analyzing latent semantic themes")

        # Extract document texts and sources
        doc_texts = [doc["processed_content"] for doc in doc_contents]
        doc_sources = [doc["source"] for doc in doc_contents]

        # Check if we have enough documents
        if len(doc_texts) < 2:
            return {
                "method": "Latent Semantic Analysis",
                "themes": [{"name": "Insufficient documents for LSA", "score": 0, "keywords": []}]
            }

        # Determine primary language
        languages = [doc["language"] for doc in doc_contents]
        primary_language = Counter(languages).most_common(1)[0][0]
        is_chinese = primary_language == "zh"

        try:
            # Configure vectorizer based on language
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.chinese_stopwords if is_chinese else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Determine number of components
            n_components = min(len(doc_texts) - 1, X.shape[1], 5)
            n_components = max(1, n_components)

            # Apply SVD to find latent semantic dimensions
            svd = TruncatedSVD(n_components=n_components)
            svd = TruncatedSVD(n_components=n_components)
            X_svd = svd.fit_transform(X)

            # Process each semantic component
            themes = []
            for i, component in enumerate(svd.components_):
                # Get top terms for this component
                max_terms = min(10, len(feature_names))
                top_term_indices = component.argsort()[-(max_terms):][::-1]
                top_terms = [feature_names[idx] for idx in top_term_indices]

                # Calculate explained variance
                explained_variance = svd.explained_variance_ratio_[i]

                # Find top documents for this theme
                theme_scores = X_svd[:, i]
                top_doc_indices = theme_scores.argsort()[::-1][:5]
                top_doc_sources = [doc_sources[idx] for idx in top_doc_indices]

                # Generate theme representation
                theme_name = f"Semantic Theme {i + 1}: {', '.join(top_terms[:3])}"

                themes.append({
                    "name": theme_name,
                    "score": round(float(explained_variance), 2),
                    "keywords": top_terms,
                    "documents": top_doc_sources,
                    "variance_explained": round(float(explained_variance) * 100, 1)
                })

            # Sort themes by variance explained
            themes.sort(key=lambda x: x["score"], reverse=True)

            return {
                "method": "Latent Semantic Analysis",
                "variance_explained": round(float(sum(svd.explained_variance_ratio_)) * 100, 1),
                "themes": themes
            }

        except Exception as e:
            debug_print(self.config, f"Error in latent semantic analysis: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())

            return {
                "method": "Latent Semantic Analysis",
                "error": str(e),
                "themes": [{"name": "Analysis Error", "score": 0, "keywords": []}]
            }

    def _analyze_document_clusters(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Cluster documents by content similarity

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Document clustering results
        """
        debug_print(self.config, "Clustering documents")

        # Extract document texts and sources
        doc_texts = [doc["processed_content"] for doc in doc_contents]
        doc_sources = [doc["source"] for doc in doc_contents]

        # Check document count
        if len(doc_texts) < 2:
            return {
                "method": "Document Clustering",
                "themes": [{"name": "Insufficient documents for clustering", "score": 0, "documents": []}]
            }

        # Determine primary language
        languages = [doc["language"] for doc in doc_contents]
        primary_language = Counter(languages).most_common(1)[0][0]
        is_chinese = primary_language == "zh"

        try:
            # Configure vectorizer based on language
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.chinese_stopwords if is_chinese else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)

            # Determine number of clusters
            n_docs = len(doc_texts)
            max_clusters = min(3, max(2, n_docs // 2))
            n_clusters = min(max_clusters, n_docs - 1)

            # Perform clustering
            kmeans = kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            # Extract cluster themes
            themes = []
            for cluster_id in range(n_clusters):
                # Get documents in this cluster
                cluster_mask = (clusters == cluster_id)
                cluster_doc_indices = [i for i, mask in enumerate(cluster_mask) if mask]

                if not cluster_doc_indices:
                    continue

                # Extract documents and sources for this cluster
                cluster_texts = [doc_texts[i] for i in cluster_doc_indices]
                cluster_sources = [doc_sources[i] for i in cluster_doc_indices]

                # Extract keywords for this cluster
                keywords = extract_keywords(
                    self.config,
                    cluster_texts,
                    language=primary_language,
                    top_n=5,
                    stopwords=self.chinese_stopwords if is_chinese else self.english_stopwords
                )

                # Create theme
                theme_name = f"Cluster: {', '.join(keywords[:3])}"

                themes.append({
                    "name": theme_name,
                    "score": len(cluster_doc_indices) / len(doc_texts),
                    "keywords": keywords,
                    "documents": cluster_sources,
                    "document_count": len(cluster_sources)
                })

            # Sort clusters by size
            themes.sort(key=lambda x: x["document_count"], reverse=True)

            return {
                "method": "Document Clustering",
                "clusters": n_clusters,
                "themes": themes
            }

        except Exception as e:
            debug_print(self.config, f"Error in document clustering: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())

            return {
                "method": "Document Clustering",
                "error": str(e),
                "themes": [{"name": "Clustering Error", "score": 0, "documents": []}]
            }