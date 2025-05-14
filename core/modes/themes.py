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

        # Initialize LLM connector factory
        try:
            from core.connectors.connector_factory import ConnectorFactory
            self.factory = ConnectorFactory(config)
            self.llm_connector = self.factory.get_llm_connector()
            debug(config, "LLM connector factory initialized for theme analysis")
        except Exception as e:
            debug(config, f"Error initializing ConnectorFactory: {str(e)}")
            self.factory = None
            self.llm_connector = None

        # Prepare Chinese and English stopwords
        self._prepare_stopwords()

        # Check for spaCy availability
        self.use_spacy = False
        if SPACY_AVAILABLE and config.get('system.use_spacy', True):
            self.use_spacy = True
            debug(config, "Using spaCy for theme and topic analysis")

        debug(config, "Theme analyzer initialized")

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

        # Create a combined stopwords dictionary
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
                try:
                    results[analysis_method] = handler(doc_contents)
                except Exception as e:
                    error(f"Error in {analysis_method} analysis: {str(e)}")
                    import traceback
                    debug(self.config, traceback.format_exc())

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
            language = doc["language"]
            docs_by_language[language].append(doc)

        # Extract keywords by language
        all_keywords = []

        # Print information about document grouping
        print(
            f"Documents grouped by language: {', '.join([f'{lang}: {len(docs)}' for lang, docs in docs_by_language.items()])}")

        # Process each language group
        for language, docs in docs_by_language.items():
            print(f"Processing {len(docs)} documents in language: {language}")

            # Extract document texts
            doc_texts = [doc["processed_content"] for doc in docs]
            doc_sources = [doc["source"] for doc in docs]

            # Extract keywords using the common function
            keywords = extract_keywords(
                self.config,
                doc_texts,
                language=language,
                top_n=30,
                stopwords=self.stopwords.get(language)
            )

            # Convert to standardized format
            keyword_objects = []
            for keyword in keywords:
                # Count document occurrences
                doc_count = sum(1 for text in doc_texts if keyword in text)
                doc_sources_list = [doc_sources[i] for i, text in enumerate(doc_texts) if keyword in text]

                # Calculate score (simple frequency)
                score = doc_count / len(docs)

                keyword_objects.append({
                    "keyword": keyword,
                    "score": float(score),
                    "documents": doc_count,
                    "doc_sources": doc_sources_list
                })

            print(f"Extracted {len(keyword_objects)} keywords for language: {language}")
            all_keywords.extend(keyword_objects)

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
            debug(self.config, f"Error in content network analysis: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())

            return {
                "method": "Content Network Analysis",
                "error": str(e),
                "themes": [{"name": "Analysis Error", "centrality": 0, "nodes": []}]
            }

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
                doc_texts = [doc.get("content", "") for doc in docs]
                doc_sources = [doc.get("source", "unknown") for doc in docs]

                # Extract entities using SpaCy
                entities = extract_entities_from_text(
                    self.config,
                    doc_texts,
                    language,
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
            # Configure vectorizer based on language
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.stopwords[primary_language] if primary_language in self.stopwords else None
            )

            # Create document-term matrix
            doc_term_matrix = vectorizer.fit_transform(doc_texts)

            # Determine number of topics dynamically
            n_topics = min(max(5, len(doc_contents) // 3), 10)

            # Initialize NMF model
            nmf_model = NMF(
                n_components=n_topics,
                random_state=42,
                max_iter=200
            )

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
                top_docs_idx = topic_contribution.argsort()[::-1][:5]
                top_docs = [doc_sources[i] for i in top_docs_idx]

                # Generate topic summary using LLM if available
                description = None
                if self.llm_connector:
                    description = self._generate_topic_description(top_words)

                # Create topic representation
                topics.append({
                    "name": f"Topic {topic_idx + 1}: {', '.join(top_words[:3])}",
                    "keywords": top_words,
                    "documents": top_docs,
                    "document_count": len([i for i, score in enumerate(topic_contribution) if score > 0.1]),
                    "score": round(float(topic.max() / topic.sum()), 2),
                    "description": description
                })

            return {
                "method": "Non-Negative Matrix Factorization",
                "language": primary_language,
                "topics": topics
            }

        except Exception as e:
            debug(self.config, f"NMF Topic Modeling Error: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())
            return {"method": "Non-Negative Matrix Factorization", "error": str(e), "themes": []}.stopwords else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)
            feature_names = vectorizer.get_feature_names_out()

            # Determine number of components
            n_components = min(len(doc_texts) - 1, X.shape[1], 5)
            n_components = max(1, n_components)

            # Apply SVD to find latent semantic dimensions
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
        try:
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
                self.stopwords[primary_language] if primary_language in self