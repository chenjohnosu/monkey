"""
Theme analysis module with enhanced content-based processing and topic modeling
Consolidated module that includes both theme analysis and topic modeling
"""

import os
import sys
import io
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation, NMF
from sklearn.cluster import KMeans

from core.engine.keywords import extract_keywords, configure_vectorizer, extract_entities_from_text

from core.engine.logging import debug, warning, info, error
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.engine.common import safe_execute
from core.language.spacy_tokenizer import get_spacy_model, SPACY_AVAILABLE
from core.engine.dependencies import require, is_available

try:
    from community import best_partition
except ImportError:
    best_partition = None

class ThemeAnalyzer:
    """Analyzes themes and topics in document content with enhanced processing capabilities"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the theme analyzer with shared components"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)

        # Check if spaCy should be used
        self.use_spacy = SPACY_AVAILABLE and config.get('system.use_spacy', True)

        # Add dependency check for scikit-learn
        self.sklearn_available = require('sklearn', 'theme analysis')

        # Initialize stopwords
        self.stopwords = {}
        self._prepare_stopwords()

        # Store language-specific stopwords in accessible attributes for convenience
        self.english_stopwords = self.stopwords.get('en', set())
        self.chinese_stopwords = self.stopwords.get('zh', set())

        # Initialize LLM connector factory
        try:
            from core.connectors.connector_factory import ConnectorFactory
            self.factory = ConnectorFactory(config)
            self.llm_connector = self.factory.get_llm_connector()
            debug(config, "LLM connector initialized for theme analysis")
        except Exception as e:
            debug(config, f"Error initializing ConnectorFactory: {str(e)}")
            self.factory = None
            self.llm_connector = None

        debug(config, "Theme analyzer initialized")

    def _prepare_stopwords(self):
        """Prepare stopwords for processing"""
        # Load using utils if available first
        if self.use_spacy:
            try:
                from core.language.spacy_tokenizer import load_stopwords
                self.english_stopwords = load_stopwords('en')
                self.chinese_stopwords = load_stopwords('zh')
                self.stopwords = {
                    'en': self.english_stopwords,
                    'zh': self.chinese_stopwords
                }
                debug(self.config, "Loaded stopwords from spaCy")
                return
            except Exception as e:
                debug(self.config, f"Error loading spaCy stopwords: {str(e)}")

        # Fall back to file loading
        self.chinese_stopwords = set()
        self.english_stopwords = set()

        # Load Chinese stopwords
        try:
            with open(os.path.join('lexicon', 'stopwords_zh.txt'), 'r', encoding='utf-8') as f:
                self.chinese_stopwords = set(line.strip() for line in f if line.strip())
            debug(self.config, f"Loaded {len(self.chinese_stopwords)} Chinese stopwords from file")
        except FileNotFoundError:
            debug(self.config, "Chinese stopwords file not found, using default set")
            self.chinese_stopwords = {"的", "了", "和", "是", "就", "都", "而", "及"}

        # Load English stopwords
        try:
            with open(os.path.join('lexicon', 'stopwords_en.txt'), 'r', encoding='utf-8') as f:
                self.english_stopwords = set(line.strip() for line in f if line.strip())
            debug(self.config, f"Loaded {len(self.english_stopwords)} English stopwords from file")
        except FileNotFoundError:
            debug(self.config, "English stopwords file not found, using default set")
            self.english_stopwords = {
                "the", "and", "a", "to", "of", "in", "is", "that", "it",
                "with", "for", "as", "was", "on", "are", "by", "this"
            }

        # Store in combined dictionary
        self.stopwords = {
            'en': self.english_stopwords,
            'zh': self.chinese_stopwords
        }

    def analyze(self, workspace: str, method: str = 'all') -> Dict[str, Any]:
        """
        Analyze themes or topics in a workspace using multiple methods

        Args:
            workspace (str): The workspace to analyze
            method (str): Analysis method ('all', 'nfm', 'net', 'key', 'lsa', 'cluster', 'lda', 'nmf')

        Returns:
            Dict: Analyzed themes/topics
        """
        debug(self.config, f"Analyzing themes in workspace '{workspace}' using method '{method}'")

        # Print keyword extraction method information
        keyword_method = self.config.get('keywords.method', 'spacy')
        max_ngram_size = self.config.get('keywords.max_ngram_size', 2)
        print(f"\nKeyword extraction: {keyword_method.upper()}, n-gram size: {max_ngram_size}")

        # Validate method (combined theme and topic methods)
        valid_methods = ['all', 'nfm', 'net', 'key', 'lsa', 'cluster', 'lda', 'nmf']
        if method not in valid_methods:
            print(f"Invalid method: {method}. Must be one of: {', '.join(valid_methods)}")
            return {}

        # Check if scikit-learn is available for methods that need it
        sklearn_methods = ['lsa', 'cluster', 'lda', 'nmf']
        if method in sklearn_methods or method == 'all':
            if not require('sklearn', 'theme analysis'):
                if method != 'all':
                    print(f"Method '{method}' requires scikit-learn, which is not available.")
                    return {
                        "method": f"{method} analysis",
                        "error": "scikit-learn not available",
                        "themes": []
                    }
                print("Some methods require scikit-learn, which is not available. These will be skipped.")

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

        # Theme analysis methods
        theme_methods = {
            'nfm': self._analyze_named_entities,
            'net': self._analyze_content_network,
            'key': self._analyze_content_keywords,
            'lsa': self._analyze_latent_semantics,
            'cluster': self._analyze_document_clusters
        }

        # Topic modeling methods
        topic_methods = {
            'lda': self._analyze_lda_topics,
            'nmf': self._analyze_nmf_topics
        }

        # Combine all methods
        all_methods = {**theme_methods, **topic_methods}

        # Run selected methods
        for analysis_method, handler in all_methods.items():
            if method in ['all', analysis_method]:
                # Skip scikit-learn based methods if not available
                if analysis_method in sklearn_methods and not self.sklearn_available and method == 'all':
                    info(f"Skipping {analysis_method} analysis - scikit-learn not available")
                    continue

                try:
                    results[analysis_method] = handler(doc_contents)
                except Exception as e:
                    error(f"Error in {analysis_method} analysis: {str(e)}")
                    import traceback
                    debug(self.config, traceback.format_exc())
                    results[analysis_method] = {
                        "method": f"{analysis_method.capitalize()} Analysis",
                        "error": str(e),
                        "themes": []
                    }

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
                debug(self.config, f"Error preprocessing document: {str(e)}")

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

    def _analyze_content_keywords(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Extract and analyze keywords from document content

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug(self.config, "Analyzing content keywords")

        # Group documents by language
        docs_by_language = defaultdict(list)
        for doc in doc_contents:
            language = doc.get("language", "en")
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
                "name": f"Keyword: {kw['keyword']}",
                "keyword": kw["keyword"],
                "score": kw["score"],
                "documents": kw["documents"],
                "keywords": [kw["keyword"]]  # Include as part of keywords array for consistency
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
        debug(self.config, "Analyzing content relationships between documents")

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
            # Check for required dependencies
            if not require('sklearn', 'cosine similarity calculation') or not require('networkx', 'network analysis'):
                return {
                    "method": "Content Network Analysis",
                    "error": "Required dependencies not available",
                    "themes": []
                }

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
            import networkx as nx
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
                # Try Louvain method if available
                if require('community', 'community detection'):
                    from community import best_partition
                    partition = best_partition(doc_network)
                else:
                    # Fallback to connected components
                    partition = {}
                    for i, component in enumerate(nx.connected_components(doc_network)):
                        for node in component:
                            partition[node] = i
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
            debug(self.config, f"Error in content network analysis: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())

            return {
                "method": "Content Network Analysis",
                "error": str(e),
                "themes": [{"name": "Analysis Error", "centrality": 0, "nodes": []}]
            }

    def _analyze_latent_semantics(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Analyze latent semantic themes using LSA/SVD

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Latent semantic analysis results
        """
        debug(self.config, "Analyzing latent semantic themes")

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

        try:
            # Check if scikit-learn is available
            if not require('sklearn', 'latent semantic analysis'):
                return {
                    "method": "Latent Semantic Analysis",
                    "error": "scikit-learn not available",
                    "themes": []
                }

            # Configure vectorizer based on language
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.stopwords[primary_language] if primary_language in self.stopwords else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Determine number of components
            n_components = min(len(doc_texts) - 1, X.shape[1], 5)
            n_components = max(1, n_components)

            # Apply SVD to find latent semantic dimensions
            from sklearn.decomposition import TruncatedSVD
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
            debug(self.config, f"Error in latent semantic analysis: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())

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
        debug(self.config, "Clustering documents")

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

        try:
            # Check if scikit-learn is available
            if not require('sklearn', 'document clustering'):
                return {
                    "method": "Document Clustering",
                    "error": "scikit-learn not available",
                    "themes": []
                }

            # Configure vectorizer based on language
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.stopwords[primary_language] if primary_language in self.stopwords else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)

            # Determine number of clusters
            n_docs = len(doc_texts)
            max_clusters = min(3, max(2, n_docs // 2))
            n_clusters = min(max_clusters, n_docs - 1)

            # Perform clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
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
                    stopwords=self.stopwords[primary_language] if primary_language in self.stopwords else None
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
            debug(self.config, f"Error in document clustering: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())

            return {
                "method": "Document Clustering",
                "error": str(e),
                "themes": [{"name": "Clustering Error", "score": 0, "documents": []}]
            }

    def _analyze_lda_topics(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Perform Latent Dirichlet Allocation topic modeling

        Args:
            doc_contents (List[Dict]): Documents to analyze

        Returns:
            Dict: LDA topic analysis results
        """
        try:
            # Check if scikit-learn is available
            if not require('sklearn', 'LDA topic modeling'):
                return {
                    "method": "Latent Dirichlet Allocation",
                    "error": "scikit-learn not available",
                    "themes": []
                }

            # Prepare documents
            doc_texts = [doc["processed_content"] for doc in doc_contents]
            doc_sources = [doc["source"] for doc in doc_contents]

            # Determine primary language
            languages = [doc["language"] for doc in doc_contents]
            primary_language = Counter(languages).most_common(1)[0][0]

            # Configure vectorization
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.stopwords[primary_language] if primary_language in self.stopwords else None
            )

            # Create document-term matrix
            doc_term_matrix = vectorizer.fit_transform(doc_texts)

            # Determine number of topics
            n_topics = min(max(5, len(doc_contents) // 3), 10)

            # Apply LDA
            from sklearn.decomposition import LatentDirichletAllocation
            lda_model = LatentDirichletAllocation(
                n_components=n_topics,
                random_state=42,
                max_iter=10
            )
            lda_output = lda_model.fit_transform(doc_term_matrix)

            # Extract topics
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for topic_idx, topic in enumerate(lda_model.components_):
                # Get top words
                top_words_idx = topic.argsort()[:-10 - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]

                # Calculate topic contribution
                topic_contribution = lda_output[:, topic_idx]
                top_docs_idx = topic_contribution.argsort()[::-1][:5]
                top_docs = [doc_sources[i] for i in top_docs_idx]

                # Generate topic description with LLM if available
                description = None
                if self.llm_connector:
                    description = self._generate_topic_description(top_words)

                topics.append({
                    "name": f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}",
                    "keywords": top_words,
                    "documents": top_docs,
                    "document_count": len([i for i, score in enumerate(topic_contribution) if score > 0.1]),
                    "score": float(topic.max() / topic.sum()),  # Convert to Python float
                    "description": description
                })

            return {
                "method": "Latent Dirichlet Allocation",
                "language": primary_language,
                "topics": topics
            }

        except Exception as e:
            debug(self.config, f"LDA Topic Modeling Error: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())
            return {"method": "Latent Dirichlet Allocation", "error": str(e), "themes": []}

    def _analyze_nmf_topics(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Perform Non-Negative Matrix Factorization topic modeling

        Args:
            doc_contents (List[Dict]): Documents to analyze

        Returns:
            Dict: NMF topic analysis results
        """
        # Check if scikit-learn is available
        if not require('sklearn', 'NMF topic modeling'):
            return {
                "method": "Non-Negative Matrix Factorization",
                "error": "scikit-learn not available",
                "themes": []
            }

        try:
            # Prepare documents
            doc_texts = [doc.get("processed_content", "") for doc in doc_contents]
            doc_sources = [doc.get("source", "") for doc in doc_contents]

            # Check if we have enough documents with content
            valid_docs = [text for text in doc_texts if text]
            if len(valid_docs) < 2:
                return {
                    "method": "Non-Negative Matrix Factorization",
                    "error": "Insufficient documents with content for analysis",
                    "themes": []
                }

            # Determine primary language
            languages = [doc.get("language", "en") for doc in doc_contents]
            primary_language = Counter(languages).most_common(1)[0][0]

            print(f"Performing NMF topic modeling on {len(valid_docs)} documents in {primary_language}")

            # Configure vectorization
            vectorizer = configure_vectorizer(
                self.config,
                len(valid_docs),
                primary_language,
                self.stopwords.get(primary_language, set())
            )

            # Create document-term matrix
            doc_term_matrix = vectorizer.fit_transform(valid_docs)

            # Determine number of topics dynamically
            n_topics = min(max(3, len(valid_docs) // 3), 10)

            # Import NMF class
            from sklearn.decomposition import NMF

            # Initialize NMF model with error handling
            try:
                nmf_model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=500  # Increased iterations for better convergence
                )
            except Exception as e:
                debug(self.config, f"Error initializing NMF with parameters: {str(e)}")
                # Try with minimal parameters if the first attempt fails
                nmf_model = NMF(n_components=n_topics)

            # Fit the model
            nmf_output = nmf_model.fit_transform(doc_term_matrix)

            # Extract feature names
            feature_names = vectorizer.get_feature_names_out()
            topics = []

            for topic_idx, topic in enumerate(nmf_model.components_):
                # Get top words for this topic
                top_words_idx = topic.argsort()[:-10 - 1:-1]
                top_words = [feature_names[i] for i in top_words_idx]

                # Calculate topic contribution to documents
                topic_contribution = nmf_output[:, topic_idx]

                # Find documents with significant contribution from this topic
                significant_docs = []
                for i, score in enumerate(topic_contribution):
                    if score > 0.1:  # Threshold for significance
                        significant_docs.append(i)

                # Get sources for top documents
                top_docs_idx = topic_contribution.argsort()[::-1][:5]
                top_docs = [doc_sources[i] for i in top_docs_idx if i < len(doc_sources)]

                # Calculate normalized score
                score = float(topic.max() / topic.sum()) if topic.sum() > 0 else 0.0

                # Generate topic name from keywords
                topic_name = f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}"

                # Generate topic description using LLM if available
                description = None
                if hasattr(self, 'llm_connector') and self.llm_connector:
                    description = self._generate_topic_description(top_words)

                # Create topic representation
                topics.append({
                    "name": topic_name,
                    "keywords": top_words,
                    "documents": top_docs,
                    "document_count": len(significant_docs),
                    "score": round(score, 2),
                    "description": description,
                    "language": primary_language  # Include language info
                })

            return {
                "method": "Non-Negative Matrix Factorization",
                "language": primary_language,
                "topics": topics,
                "num_topics": n_topics
            }

        except Exception as e:
            debug(self.config, f"NMF Topic Modeling Error: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())
            return {
                "method": "Non-Negative Matrix Factorization",
                "error": str(e),
                "themes": []
            }

    def _generate_topic_description(self, keywords):
        """
        Generate a description for a topic using LLM

        Args:
            keywords (List[str]): Keywords representing the topic

        Returns:
            str: Generated description or None if not available
        """
        if not hasattr(self, 'llm_connector') or not self.llm_connector:
            return None

        try:
            prompt = f"""Analyze these keywords and provide a concise topic description (1-2 sentences):
            Keywords: {', '.join(keywords[:10])}

            What conceptual area do these words represent? Provide a specific but concise explanation."""

            model = self.config.get('llm.default_model', 'mistral')
            description = self.llm_connector.generate(
                prompt,
                model=model,
                max_tokens=100
            )
            return description.strip()
        except Exception as e:
            debug(self.config, f"Error generating topic description: {str(e)}")
            return None

    def _extract_chinese_keywords(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract keywords from Chinese document content using spaCy

        Args:
            docs (List[Dict]): Chinese documents

        Returns:
            List[Dict]: Extracted keywords
        """
        # Log extraction attempt
        print(f"Extracting keywords from {len(docs)} Chinese documents")

        # Check if documents contain content
        if not docs or all(not doc.get("content") for doc in docs):
            print("No valid content found in Chinese documents")
            return []

        try:
            # Get spaCy model if available
            if self.use_spacy:
                from core.language.spacy_tokenizer import get_spacy_model
                nlp = get_spacy_model('zh')

                if nlp:
                    print("Using spaCy for Chinese keyword extraction")

                    # Extract all texts for processing
                    all_texts = [doc["content"] for doc in docs if doc.get("content")]
                    combined_text = "".join(all_texts)

                    # Process with spaCy
                    processed_doc = nlp(combined_text[:50000])  # Limit size for memory

                    # Extract noun phrases and entities as potential keywords
                    keywords = []

                    # Track frequencies
                    token_freq = Counter()

                    # Count token frequencies, focusing on nouns and proper nouns
                    for token in processed_doc:
                        if not token.is_stop and not token.is_punct and len(token.text) >= 2:
                            # Give higher weight to nouns and named entities
                            if token.pos_ in ('NOUN', 'PROPN'):
                                token_freq[token.text] += 2
                            else:
                                token_freq[token.text] += 1

                    # Add named entities with high weight
                    for ent in processed_doc.ents:
                        if len(ent.text) >= 2:
                            token_freq[ent.text] += 3

                    # Extract top tokens by frequency
                    for token, count in token_freq.most_common(30):
                        # Count document occurrences
                        doc_count = sum(1 for doc in docs if token in doc.get("content", ""))

                        # Skip tokens that appear in too few documents
                        if doc_count < 2 and len(docs) > 3:
                            continue

                        # Calculate score
                        score = count / sum(token_freq.values()) if token_freq else 0

                        # Add to keywords
                        keywords.append({
                            "keyword": token,
                            "score": float(score),
                            "documents": doc_count,
                            "doc_sources": [doc["source"] for doc in docs if token in doc.get("content", "")]
                        })

                    print(f"Extracted {len(keywords)} keywords using spaCy")
                    return keywords

            # Fall back to character-based n-gram extraction
            # Count frequencies of character and word n-grams
            character_counts = Counter()
            bigram_counts = Counter()
            trigram_counts = Counter()

            # Document sources for tracking
            doc_sources = [doc["source"] for doc in docs]

            # Count frequencies
            for doc in docs:
                text = doc.get("content", "")
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
                text = doc.get("content", "")
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
                if bigram in self.stopwords.get('zh', set()):
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
                if all(char in self.stopwords.get('zh', set()) for char in trigram):
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

            # Sort keywords by score
            keywords.sort(key=lambda x: x["score"], reverse=True)

            # Return top keywords
            return keywords[:30]  # Return more keywords to ensure we have enough after filtering

        except Exception as e:
            print(f"Error extracting Chinese keywords: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return []

    def _extract_english_keywords(self, docs: List[Dict]) -> List[Dict]:
        """
        Extract keywords from English document content with improved error handling

        Args:
            docs (List[Dict]): English documents

        Returns:
            List[Dict]: Extracted keywords
        """
        # Extract document texts
        doc_texts = [doc.get("content", "") for doc in docs]
        doc_sources = [doc.get("source", "") for doc in docs]

        # Log extraction attempt
        print(f"Extracting keywords from {len(doc_texts)} English documents")

        # Check if documents contain content
        if not doc_texts or all(not text for text in doc_texts):
            print("No valid content found in English documents")
            return []

        # Try spaCy-based extraction first
        if self.use_spacy:
            try:
                from core.language.spacy_tokenizer import get_spacy_model
                nlp = get_spacy_model('en')

                if nlp:
                    print("Using spaCy for English keyword extraction")

                    # Create mapping to store document sources for each keyword
                    keyword_doc_mapping = defaultdict(list)

                    # Combine texts for efficient processing, with limits to avoid memory issues
                    combined_text = " ".join(doc_texts)
                    if len(combined_text) > 100000:
                        combined_text = combined_text[:100000]  # Reasonable limit

                    # Process with spaCy
                    doc = nlp(combined_text)

                    # Count noun phrases, entities, and important tokens
                    counts = Counter()

                    # Add noun chunks (noun phrases)
                    for chunk in doc.noun_chunks:
                        # Clean up the phrase
                        clean_phrase = " ".join([token.lemma_ for token in chunk
                                                 if not token.is_stop and not token.is_punct])
                        if clean_phrase and len(clean_phrase) > 3:
                            counts[clean_phrase] += 2  # Give more weight to noun phrases

                    # Add named entities
                    for ent in doc.ents:
                        if len(ent.text) > 2:
                            counts[ent.text] += 3  # Give even more weight to entities

                    # Add important tokens (nouns, verbs, adjectives)
                    for token in doc:
                        if token.pos_ in ['NOUN', 'PROPN', 'VERB', 'ADJ'] and not token.is_stop:
                            counts[token.lemma_] += 1

                    # Calculate document occurrence for each term
                    for term in counts:
                        for doc_idx, doc_text in enumerate(doc_texts):
                            if doc_text and term.lower() in doc_text.lower():
                                keyword_doc_mapping[term].append(doc_sources[doc_idx])

                    # Create keyword objects
                    keywords = []
                    for term, count in counts.most_common(50):  # Get more candidates
                        # Skip very short terms and terms in too few documents
                        if len(term) < 3:
                            continue

                        doc_sources_list = list(set(keyword_doc_mapping[term]))

                        # For small document sets, accept terms in just one document
                        min_docs = 2 if len(docs) > 4 else 1
                        if len(doc_sources_list) < min_docs:
                            continue

                        # Calculate term score based on frequency and document coverage
                        score = count / sum(counts.values()) if counts else 0

                        keywords.append({
                            "keyword": term,
                            "score": float(score),
                            "documents": len(doc_sources_list),
                            "doc_sources": doc_sources_list
                        })

                    # Keep top keywords
                    return keywords[:30]
            except Exception as e:
                print(f"Error in spaCy keyword extraction: {str(e)}")
                # Fall back to TF-IDF

        # Fall back to TF-IDF based extraction if spaCy unavailable or errors
        try:
            # Check if scikit-learn is available
            if not require('sklearn', 'TF-IDF keyword extraction'):
                print("scikit-learn not available for TF-IDF keyword extraction")
                # Return minimal results rather than nothing
                return [{"keyword": "extraction-unavailable", "score": 0.0, "documents": 0, "doc_sources": []}]

            # Use TF-IDF vectorizer to identify important terms
            from sklearn.feature_extraction.text import TfidfVectorizer

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
            keyword_doc_mapping = defaultdict(list)

            for doc_id, doc_text in enumerate(doc_texts):
                if not doc_text:
                    continue

                # Check each term in feature names
                for term in feature_names:
                    if term in doc_text.lower():
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

    def _output_results(self, workspace: str, results: Dict, method: str):
        """
        Output theme analysis results using output manager

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

            print(f"\nFound {len(themes)} themes/topics")

            for theme in themes:
                # Print theme name
                name = theme.get('name', 'Unnamed Theme')
                self.output_manager.print_formatted('mini_header', name)

                # Print theme keywords
                if 'keywords' in theme and theme['keywords']:
                    self.output_manager.print_formatted('kv', ', '.join(theme['keywords']), key="Keywords")
                elif 'keyword' in theme:
                    self.output_manager.print_formatted('kv', theme['keyword'], key="Keyword")

                # Print various metrics
                metrics = [
                    ('score', 'Score'),
                    ('frequency', 'Frequency'),
                    ('centrality', 'Centrality'),
                    ('document_count', 'Documents'),
                    ('documents', 'Documents')
                ]

                for key, label in metrics:
                    if key in theme and theme[key] is not None:
                        self.output_manager.print_formatted('kv', theme[key], key=label)

                # Print topic/theme type if available
                if 'type' in theme:
                    self.output_manager.print_formatted('kv', theme['type'], key="Type")

                # Print document sources
                if 'documents' in theme and isinstance(theme['documents'], list) and theme['documents']:
                    print("\n  Document sources:")
                    for doc in theme['documents'][:5]:
                        self.output_manager.print_formatted('list', str(doc), indent=4)

                    if len(theme['documents']) > 5:
                        print(f"  ... and {len(theme['documents']) - 5} more")

                # Print documents list if available
                if 'documents_list' in theme and isinstance(theme['documents_list'], list) and theme['documents_list']:
                    print("\n  Document sources:")
                    for doc in theme['documents_list'][:5]:
                        self.output_manager.print_formatted('list', str(doc), indent=4)

                    if len(theme['documents_list']) > 5:
                        print(f"  ... and {len(theme['documents_list']) - 5} more")

                # Print description if available
                if 'description' in theme and theme['description']:
                    print("\n  Description:")
                    print(f"  {theme['description']}")

        # Save results to file
        output_format = self.config.get('system.output_format', 'txt')
        filepath = self.output_manager.save_theme_analysis(workspace, results, method, output_format)

        # Show success message
        self.output_manager.print_formatted('feedback', f"Results saved to: {filepath}")

    def _analyze_named_entities(self, doc_contents: List[Dict]) -> Dict[str, Any]:
        """
        Extract and analyze named entities from document content using SpaCy

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Named entity analysis results
        """
        # Print detailed info about what's being processed
        print(f"Processing {len(doc_contents)} documents for named entity analysis")

        # Group documents by language
        docs_by_language = defaultdict(list)
        for doc in doc_contents:
            language = doc.get("language", "en")
            docs_by_language[language].append(doc)

        # Prepare results
        all_entities = []

        # Process entities for each language
        for language, docs in docs_by_language.items():
            print(f"\nProcessing {language} language documents. Total: {len(docs)}")

            try:
                # Extract entities using the shared extraction function
                doc_texts = [doc.get("content", "") for doc in docs]  # Use original content for better entity detection
                doc_sources = [doc.get("source", "unknown") for doc in docs]

                if language == 'zh':
                    print(f"Using specialized Chinese entity extraction")

                # Extract entities using the enhanced function from core/engine/keywords.py
                entities = extract_entities_from_text(
                    self.config,
                    doc_texts,
                    language,
                    top_n=50,  # Extract more entities initially
                    stopwords=self.stopwords[language] if language in self.stopwords else None
                )

                print(f"Extracted {len(entities)} entities for {language}")

                # Map entities to documents
                for entity in entities:
                    entity_text = entity['text']
                    doc_count = 0
                    entity_sources = []

                    for idx, text in enumerate(doc_texts):
                        if entity_text in text:
                            doc_count += 1
                            entity_sources.append(doc_sources[idx])

                    # Only add entities that appear in documents
                    if doc_count > 0:
                        all_entities.append({
                            'value': entity_text,
                            'count': entity['count'],
                            'documents': doc_count,
                            'type': entity['type'],
                            'sources': entity_sources
                        })

            except Exception as e:
                print(f"Error extracting entities for {language}: {str(e)}")
                import traceback
                traceback.print_exc()

        # Sort entities by frequency and document count
        if all_entities:
            sorted_entities = sorted(all_entities, key=lambda x: (x['documents'], x['count']), reverse=True)

            # Take top entities (more than before)
            top_entities = sorted_entities[:100]  # Take more entities

            print(f"\nTotal unique entities found: {len(sorted_entities)}")
            print("Top 10 entities:")
            for i, entity in enumerate(top_entities[:10]):
                print(f"  {entity['value']}: count={entity['count']}, documents={entity['documents']}")

            # Generate themes
            themes = []
            for entity in top_entities[:30]:  # Generate more themes
                themes.append({
                    'name': f"Theme: {entity['value']}",
                    'keywords': [entity['value']],
                    'frequency': entity['count'],
                    'document_count': entity['documents'],
                    'type': entity.get('type', 'UNKNOWN')
                })

            # Return results with all entities counted as significant
            return {
                "method": "Named Entity Analysis",
                "entity_count": len(sorted_entities),
                "significant_entities": len(sorted_entities),  # Count all as significant
                "themes": themes
            }
        else:
            print("No entities found")
            return {
                "method": "Named Entity Analysis",
                "entity_count": 0,
                "significant_entities": 0,
                "themes": []
            }

    