"""
Theme analysis module with enhanced content-based processing
"""

import os
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from core.engine.utils import debug_print, ensure_dir
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor

# Import jieba for Chinese text segmentation if available
try:
    import jieba

    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    print("jieba not available, falling back to character-based tokenization for Chinese")


class ChineseTokenizer:
    """Tokenizer for Chinese text using jieba if available"""

    def __init__(self):
        self.use_jieba = JIEBA_AVAILABLE

    def __call__(self, text):
        """Tokenize Chinese text"""
        if not text:
            return []

        if self.use_jieba:
            # Use jieba for word segmentation
            return list(jieba.cut(text))
        else:
            # Fallback to character-based tokenization
            return [char for char in text if '\u4e00' <= char <= '\u9fff']

class ThemeAnalyzer:
    """Analyzes document themes with a focus on content semantics"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the theme analyzer"""
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

        # Initialize tokenizer if available
        self.tokenizer = None
        try:
            from core.language.tokenizer import Tokenizer
            self.tokenizer = Tokenizer(config)
            debug_print(config, "Tokenizer initialized")
        except ImportError:
            debug_print(config, "Tokenizer not available, using simplified processing")

        # Initialize keyword document mappings
        self.keyword_doc_mapping = {}

        # Load Chinese stopwords if available
        self.chinese_stopwords = set()
        try:
            with open('stopwords_zh.txt', 'r', encoding='utf-8') as file:
                self.chinese_stopwords = set(line.strip() for line in file if line.strip())
            debug_print(config, f"Loaded {len(self.chinese_stopwords)} Chinese stopwords")
        except FileNotFoundError:
            debug_print(config, "Chinese stopwords file not found")

        # Load English stopwords if available
        self.english_stopwords = set()
        try:
            with open('stopwords_en.txt', 'r', encoding='utf-8') as file:
                self.english_stopwords = set(line.strip() for line in file if line.strip())
            debug_print(config, f"Loaded {len(self.english_stopwords)} English stopwords")
        except FileNotFoundError:
            # Fallback to a basic set of English stopwords
            self.english_stopwords = self._get_english_stopwords()
            debug_print(config, "English stopwords file not found, using default set")

        # Check for jieba availability for Chinese text segmentation
        try:
            import jieba
            self.jieba_available = True
            debug_print(config, "jieba available for Chinese word segmentation")
        except ImportError:
            self.jieba_available = False
            debug_print(config, "jieba not available, falling back to character-based segmentation for Chinese")

        # Import specialized analysis libraries if available
        try:
            import networkx as nx
            self.nx_available = True
        except ImportError:
            self.nx_available = False
            debug_print(config, "networkx not available, some network analysis features will be limited")

        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.decomposition import TruncatedSVD
            from sklearn.cluster import KMeans
            self.sklearn_available = True
        except ImportError:
            self.sklearn_available = False
            debug_print(config, "scikit-learn not available, some analysis features will be limited")

        debug_print(config, "Enhanced theme analyzer initialized")

    def analyze(self, workspace, method='all'):
        """
        Analyze themes in a workspace using content-based processing

        Args:
            workspace (str): The workspace to analyze
            method (str): Analysis method ('all', 'nfm', 'net', 'key', 'lsa', 'cluster')
        """
        debug_print(self.config, f"Analyzing themes in workspace '{workspace}' using method '{method}'")

        # Validate method
        valid_methods = ['all', 'nfm', 'net', 'key', 'lsa', 'cluster']
        if method not in valid_methods:
            print(f"Invalid method: {method}. Must be one of: {', '.join(valid_methods)}")
            return

        # Check if workspace exists
        data_dir = os.path.join("data", workspace)
        if not os.path.exists(data_dir):
            print(f"Workspace '{workspace}' does not exist or has no vector store")
            return

        # Get documents from vector store
        docs = self.storage_manager.get_documents(workspace)
        if not docs:
            print(f"No documents found in workspace '{workspace}'")
            return

        print(f"Analyzing {len(docs)} documents in workspace '{workspace}'")

        # Extract and preprocess document content for analysis
        doc_contents, doc_languages = self._extract_document_contents(docs)

        # Display language distribution
        self._display_language_stats(doc_languages)

        # Run selected analysis methods
        results = {}

        if method in ['all', 'nfm']:
            print("\nRunning Named Entity Analysis...")
            results['nfm'] = self._analyze_named_entities(doc_contents)

        if method in ['all', 'net']:
            print("\nRunning Content Network Analysis...")
            results['net'] = self._analyze_content_network(doc_contents)

        if method in ['all', 'key']:
            print("\nRunning Keyword Extraction...")
            results['key'] = self._analyze_content_keywords(doc_contents)

        if method in ['all', 'lsa']:
            print("\nRunning Latent Semantic Analysis...")
            results['lsa'] = self._analyze_latent_semantics(doc_contents)

        if method in ['all', 'cluster']:
            print("\nRunning Document Clustering...")
            results['cluster'] = self._analyze_document_clusters(doc_contents)

        # Output results
        self._output_results(workspace, results, method)

    def _extract_document_contents(self, docs):
        """
        Extract content from documents and preprocess for analysis

        Args:
            docs (List[Dict]): Raw documents

        Returns:
            Tuple: (processed documents, language counts)
        """
        debug_print(self.config, "Extracting and preprocessing document content")

        doc_contents = []
        doc_languages = Counter()

        for doc in docs:
            # Extract content and metadata
            content = doc.get("content", "")
            source = doc.get("metadata", {}).get("source", "unknown")
            language = doc.get("metadata", {}).get("language", "en")

            # Track language
            doc_languages[language] += 1

            # Use processed content if available or process raw content
            processed_content = doc.get("processed_content", "")
            if not processed_content and content:
                try:
                    processed = self.text_processor.preprocess(content)
                    processed_content = processed["processed"]
                except Exception as e:
                    debug_print(self.config, f"Error preprocessing document: {str(e)}")
                    processed_content = content

            # Ensure we have content to analyze
            if not processed_content and not content:
                debug_print(self.config, f"Skipping document with no content: {source}")
                continue

            # Use processed content as primary, with raw content as fallback
            text_to_analyze = processed_content if processed_content else content

            # Add document to dataset
            doc_contents.append({
                "id": len(doc_contents),
                "source": source,
                "language": language,
                "content": content,
                "processed_content": text_to_analyze  # This is what we'll analyze
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

    def _analyze_named_entities(self, doc_contents):
        """
        Extract and analyze named entities from document content

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug_print(self.config, "Analyzing named entities in document content")

        # Extract entities from documents based on language
        entities_by_doc = {}
        all_entities = Counter()

        for doc in doc_contents:
            doc_id = doc["id"]
            language = doc["language"]
            text = doc["processed_content"]

            # Extract entities using appropriate method for language
            if language == "zh":
                doc_entities = self._extract_chinese_entities(text)
            else:
                doc_entities = self._extract_english_entities(text)

            # Add to entity counts
            entity_counts = Counter(doc_entities)
            entities_by_doc[doc_id] = entity_counts
            all_entities.update(entity_counts)

        # Filter significant entities (appear in multiple docs or high frequency)
        significant_entities = {}

        for entity, count in all_entities.items():
            # Count documents containing this entity
            doc_count = sum(1 for doc_entities in entities_by_doc.values()
                            if entity in doc_entities)

            # Entity is significant if it appears in multiple docs or has high frequency
            if doc_count > 1 or count > 5:
                significant_entities[entity] = {
                    "count": count,
                    "doc_count": doc_count,
                    "docs": [doc_id for doc_id, doc_entities in entities_by_doc.items()
                             if entity in doc_entities]
                }

        # Build entity co-occurrence network
        entity_network = nx.Graph()

        # Add nodes for significant entities
        for entity, data in significant_entities.items():
            entity_network.add_node(entity,
                                   count=data["count"],
                                   doc_count=data["doc_count"])

        # Add edges for entities that co-occur in documents
        for entity1, data1 in significant_entities.items():
            docs1 = set(data1["docs"])

            for entity2, data2 in significant_entities.items():
                if entity1 >= entity2:  # Skip self-loops and duplicates
                    continue

                docs2 = set(data2["docs"])
                common_docs = docs1.intersection(docs2)

                if common_docs:
                    # Calculate Jaccard similarity
                    similarity = len(common_docs) / len(docs1.union(docs2))

                    if similarity > 0.1:  # Only add significant connections
                        entity_network.add_edge(entity1, entity2,
                                              weight=similarity,
                                              common_docs=len(common_docs))

        # Find entity communities
        try:
            from community import best_partition
            partition = best_partition(entity_network)
        except ImportError:
            # Fall back to connected components
            partition = {}
            for i, component in enumerate(nx.connected_components(entity_network)):
                for node in component:
                    partition[node] = i

        # Group entities by community
        communities = defaultdict(list)
        for entity, community_id in partition.items():
            communities[community_id].append(entity)

        # Identify theme topics and format results
        themes = []
        for community_id, entity_list in communities.items():
            # Skip communities with only one entity
            if len(entity_list) < 2:
                continue

            # Sort entities by frequency
            entities_sorted = sorted(
                entity_list,
                key=lambda e: significant_entities[e]["count"],
                reverse=True
            )

            # Create theme
            top_entities = entities_sorted[:3]
            theme_name = " / ".join(top_entities)

            # Calculate theme score (average entity frequency)
            theme_score = sum(significant_entities[e]["count"] for e in entity_list) / len(entity_list)
            normalized_score = min(1.0, theme_score / max(all_entities.values()))

            # Count documents in this theme
            theme_docs = set()
            for entity in entity_list:
                theme_docs.update(significant_entities[entity]["docs"])

            themes.append({
                "name": f"Theme: {theme_name}",
                "frequency": round(normalized_score, 2),
                "keywords": entities_sorted[:5],
                "entity_count": len(entity_list),
                "document_count": len(theme_docs)
            })

        # Sort themes by document count and frequency
        themes.sort(key=lambda x: (x["document_count"], x["frequency"]), reverse=True)

        return {
            "method": "Named Entity Analysis",
            "entity_count": len(all_entities),
            "significant_entities": len(significant_entities),
            "themes": themes
        }

    def _extract_english_entities(self, text):
        """
        Extract named entities from English text

        Args:
            text (str): English text

        Returns:
            List[str]: Extracted entities
        """
        # More sophisticated entity extraction
        import re

        # Detect noun phrases (simplified approach)
        # Look for sequences of capitalized words
        capitalized_phrases = re.findall(r'\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b', text)

        # Look for important terms with specific patterns
        technical_terms = re.findall(r'\b[A-Za-z]+-?[A-Za-z]+\b', text)

        # Filter and combine entities
        entities = []

        # Add capitalized phrases (potential named entities)
        for phrase in capitalized_phrases:
            if len(phrase) > 3 and phrase.lower() not in self._get_english_stopwords():
                entities.append(phrase)

        # Add important technical terms
        for term in technical_terms:
            if len(term) > 5 and term.lower() not in self._get_english_stopwords():
                if re.search(r'[A-Z]', term) and re.search(r'[a-z]', term):  # CamelCase or mixedCase
                    entities.append(term)

        return entities

    def _extract_chinese_entities(self, text):
        """
        Extract named entities from Chinese text

        Args:
            text (str): Chinese text

        Returns:
            List[str]: Extracted entities
        """
        # Extract Chinese entity candidates
        import re

        # Focus on longer character sequences (more likely to be meaningful entities)
        entities = []

        # Extract sequences of 2-4 Chinese characters
        for length in range(2, 5):
            pattern = r'[\u4e00-\u9fff]{' + str(length) + '}'
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return entities

    def _get_english_stopwords(self):
        """Get English stopwords"""
        return {
            "the", "and", "a", "to", "of", "in", "is", "that", "it", "with", "for", "as", "was",
            "on", "are", "by", "this", "be", "from", "an", "or", "have", "had", "has", "were",
            "which", "not", "at", "but", "when", "if", "they", "their", "there", "one", "all"
        }

    def _analyze_content_network(self, doc_contents):
        """
        Analyze content relationships between documents

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug_print(self.config, "Analyzing content relationships between documents")

        # Extract document text and metadata
        doc_texts = []
        doc_metadata = []

        for doc in doc_contents:
            doc_texts.append(doc["processed_content"])
            doc_metadata.append({
                "id": doc["id"],
                "source": doc["source"],
                "language": doc["language"]
            })

        # Check if we have enough documents
        if len(doc_texts) < 2:
            return {
                "method": "Content Network Analysis",
                "themes": [{"name": "Insufficient documents for analysis", "centrality": 0, "nodes": []}]
            }

        # Determine primary language
        languages = [meta["language"] for meta in doc_metadata]
        primary_language = Counter(languages).most_common(1)[0][0]
        is_chinese = primary_language == "zh"

        # Create TF-IDF vectors for documents with parameters appropriate for corpus size
        try:
            # Configure vectorizer for language and corpus size using utility function
            from core.engine.utils import configure_vectorizer
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                primary_language,
                self.chinese_stopwords if is_chinese else None
            )

            # Generate document vectors
            tfidf_matrix = vectorizer.fit_transform(doc_texts)

            # Calculate document similarity matrix
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
                from community import best_partition
                partition = best_partition(doc_network)
            except ImportError:
                # Fall back to connected components
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
                if len(doc_ids) < 1:  # Accept smaller communities for small document sets
                    continue

                # Extract theme keywords from community documents
                community_texts = [doc_texts[doc_id] for doc_id in doc_ids]
                keywords = self._extract_community_keywords(community_texts, is_chinese)

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

    def _extract_community_keywords(self, texts, is_chinese=False, top_n=5):
        """
        Extract keywords that characterize a community of texts

        Args:
            texts (List[str]): Texts in the community
            is_chinese (bool): Whether the texts are in Chinese
            top_n (int): Number of keywords to extract

        Returns:
            List[str]: Keywords for the community
        """
        try:
            # Skip if no texts
            if not texts:
                return ["No text available"]

            # Combine texts for this community
            combined_text = " ".join(texts)

            # Set appropriate parameters for small document sets
            if is_chinese:
                # For Chinese, use character-based tokenization
                vectorizer = TfidfVectorizer(
                    min_df=1,
                    max_df=1.0,
                    stop_words=None,
                    tokenizer=ChineseTokenizer()
                )
            else:
                vectorizer = TfidfVectorizer(
                    min_df=1,
                    max_df=1.0,
                    stop_words="english"
                )

            # Create a corpus with just the combined text
            corpus = [combined_text]

            # Fit vectorizer and extract top terms
            X = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()

            # Get scores for the community text
            scores = np.asarray(X[0].sum(axis=0)).ravel()

            # Sort terms by score
            scored_terms = list(zip(feature_names, scores))
            sorted_terms = sorted(scored_terms, key=lambda x: x[1], reverse=True)

            # Filter stopwords for Chinese if needed
            if is_chinese and self.chinese_stopwords:
                sorted_terms = [(term, score) for term, score in sorted_terms
                                if term not in self.chinese_stopwords]

            # Get top terms
            keywords = [term for term, score in sorted_terms[:top_n]]

            return keywords

        except Exception as e:
            debug_print(self.config, f"Error extracting community keywords: {str(e)}")

            # Fall back to simple word counting
            from collections import Counter

            if is_chinese:
                # For Chinese, use character-based counting
                if JIEBA_AVAILABLE:
                    words = []
                    for text in texts:
                        words.extend(jieba.cut(text))
                else:
                    words = []
                    for text in texts:
                        words.extend([char for char in text if '\u4e00' <= char <= '\u9fff'])
            else:
                words = " ".join(texts).split()

            word_counts = Counter(words)

            # Filter stopwords
            if is_chinese:
                filtered_words = [word for word, _ in word_counts.most_common(20)
                                  if word not in self.chinese_stopwords and len(word) > 0]
            else:
                stopwords = self._get_english_stopwords()
                filtered_words = [word for word, _ in word_counts.most_common(20)
                                  if word.lower() not in stopwords and len(word) > 3]

            return filtered_words[:top_n] if filtered_words else ["No significant keywords found"]

    def _analyze_content_keywords(self, doc_contents):
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

        for language, docs in docs_by_language.items():
            if language == "zh":
                keywords = self._extract_chinese_content_keywords(docs)
            else:
                keywords = self._extract_english_content_keywords(docs)

            all_keywords.extend(keywords)

        # Sort keywords by score
        all_keywords.sort(key=lambda x: x["score"], reverse=True)

        # Take top keywords
        top_keywords = all_keywords[:20]

        return {
            "method": "Content Keyword Analysis",
            "themes": top_keywords
        }

    def _extract_english_content_keywords(self, docs):
        """
        Extract keywords from English document content

        Args:
            docs (List[Dict]): English documents

        Returns:
            List[Dict]: Extracted keywords
        """
        # Extract document texts and sources
        doc_texts = [doc["processed_content"] for doc in docs]
        doc_sources = [doc["source"] for doc in docs]  # Get source paths

        # Create a mapping to store document sources for each keyword
        self.keyword_doc_mapping = {}

        try:
            # Use TF-IDF vectorizer to identify important terms
            vectorizer = TfidfVectorizer(
                min_df=2,
                max_df=0.85,
                stop_words="english"
            )

            # Fit TF-IDF on documents
            tfidf_matrix = vectorizer.fit_transform(doc_texts)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Calculate average TF-IDF scores across documents
            avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).ravel()

            # Count documents containing each term and track which documents contain each term
            term_doc_counts = defaultdict(int)
            term_doc_sources = defaultdict(list)  # To track source documents for each term

            for doc_id, doc_text in enumerate(doc_texts):
                for term in set(doc_text.split()):
                    if term in feature_names:
                        term_doc_counts[term] += 1
                        term_doc_sources[term].append(doc_sources[doc_id])  # Store source path

            # Store the mapping for use in _output_results
            self.keyword_doc_mapping = term_doc_sources

            # Create keyword list
            keywords = []

            for i, term in enumerate(feature_names):
                # Skip very short terms
                if len(term) < 3:
                    continue

                score = avg_scores[i]
                doc_count = term_doc_counts.get(term, 0)

                keywords.append({
                    "keyword": term,
                    "score": round(float(score), 2),
                    "documents": doc_count,
                    "doc_sources": term_doc_sources.get(term, [])  # Store the list of source documents
                })

            # Sort by score and document count
            keywords.sort(key=lambda x: (x["score"], x["documents"]), reverse=True)

            return keywords[:20]

        except Exception as e:
            debug_print(self.config, f"Error extracting English keywords: {str(e)}")

            # Fall back to simple word frequency
            all_text = " ".join(doc_texts)
            words = all_text.split()

            # Count word frequencies
            word_counts = Counter(words)

            # Filter stopwords
            stopwords = self._get_english_stopwords()
            filtered_words = [word for word, count in word_counts.most_common(50)
                              if word.lower() not in stopwords and len(word) > 3]

            # Count documents containing each word and track source documents
            word_doc_counts = defaultdict(int)
            word_doc_sources = defaultdict(list)  # To track source documents

            for doc_id, doc_text in enumerate(doc_texts):
                for word in set(doc_text.split()):
                    if word in filtered_words:
                        word_doc_counts[word] += 1
                        word_doc_sources[word].append(doc_sources[doc_id])  # Store source path

            # Store the mapping for use in _output_results
            self.keyword_doc_mapping = word_doc_sources

            # Create keyword list
            keywords = []

            for word in filtered_words[:20]:
                count = word_counts[word]
                score = count / len(words)

                keywords.append({
                    "keyword": word,
                    "score": round(score, 2),
                    "documents": word_doc_counts.get(word, 0),
                    "doc_sources": word_doc_sources.get(word, [])  # Store the list of source documents
                })

            return keywords

    def _extract_chinese_content_keywords(self, docs):
        """
        Extract keywords from Chinese document content

        Args:
            docs (List[Dict]): Chinese documents

        Returns:
            List[Dict]: Extracted keywords
        """
        # Extract document texts and sources
        doc_texts = [doc["processed_content"] for doc in docs]
        doc_sources = [doc["source"] for doc in docs]  # Get source paths

        # Create a mapping to store document sources for each keyword
        self.keyword_doc_mapping = {}

        # For Chinese, use character n-grams
        character_counts = Counter()
        bigram_counts = Counter()
        trigram_counts = Counter()

        # Count frequencies
        for doc_text in doc_texts:
            # Count characters
            character_counts.update(doc_text)

            # Count n-grams
            for i in range(len(doc_text) - 1):
                if i < len(doc_text) - 1:
                    bigram = doc_text[i:i + 2]
                    bigram_counts[bigram] += 1

                if i < len(doc_text) - 2:
                    trigram = doc_text[i:i + 3]
                    trigram_counts[trigram] += 1

        # Count documents containing each n-gram and track source documents
        bigram_doc_counts = defaultdict(int)
        bigram_doc_sources = defaultdict(list)  # Track source documents
        trigram_doc_counts = defaultdict(int)
        trigram_doc_sources = defaultdict(list)  # Track source documents

        for doc_id, doc_text in enumerate(doc_texts):
            # Count bigrams
            doc_bigrams = set()
            for i in range(len(doc_text) - 1):
                if i < len(doc_text) - 1:
                    bigram = doc_text[i:i + 2]
                    doc_bigrams.add(bigram)

            # Update counts and sources for bigrams in this document
            for bigram in doc_bigrams:
                bigram_doc_counts[bigram] += 1
                bigram_doc_sources[bigram].append(doc_sources[doc_id])

            # Count trigrams
            doc_trigrams = set()
            for i in range(len(doc_text) - 2):
                trigram = doc_text[i:i + 3]
                doc_trigrams.add(trigram)

            # Update counts and sources for trigrams in this document
            for trigram in doc_trigrams:
                trigram_doc_counts[trigram] += 1
                trigram_doc_sources[trigram].append(doc_sources[doc_id])

        # Combine source mappings for use in _output_results
        self.keyword_doc_mapping = {}
        for bigram, sources in bigram_doc_sources.items():
            self.keyword_doc_mapping[bigram] = sources
        for trigram, sources in trigram_doc_sources.items():
            self.keyword_doc_mapping[trigram] = sources

        # Create keyword list
        keywords = []

        # Add top bigrams
        for bigram, count in bigram_counts.most_common(10):
            if count < 2:
                continue

            score = count / sum(bigram_counts.values())

            keywords.append({
                "keyword": bigram,
                "score": round(score, 2),
                "documents": bigram_doc_counts.get(bigram, 0),
                "doc_sources": bigram_doc_sources.get(bigram, [])  # Store source documents
            })

        # Add top trigrams
        for trigram, count in trigram_counts.most_common(10):
            if count < 2:
                continue

            score = count / sum(trigram_counts.values())

            keywords.append({
                "keyword": trigram,
                "score": round(score, 2),
                "documents": trigram_doc_counts.get(trigram, 0),
                "doc_sources": trigram_doc_sources.get(trigram, [])  # Store source documents
            })

        # Sort keywords by score
        keywords.sort(key=lambda x: x["score"], reverse=True)

        return keywords[:20]

    def _analyze_latent_semantics(self, doc_contents):
        """
        Analyze latent semantic themes using LSA/SVD with interpretive summaries

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug_print(self.config, "Analyzing latent semantic themes")

        # Extract document texts
        doc_texts = [doc["processed_content"] for doc in doc_contents]
        doc_sources = [doc["source"] for doc in doc_contents]

        # Get workspace name from config for accessing original files
        workspace = self.config.get('workspace.default')

        # Check if we have enough documents
        if len(doc_texts) < 2:
            return {
                "method": "Latent Semantic Analysis",
                "themes": [{"name": "Insufficient documents for LSA", "score": 0, "keywords": []}]
            }

        # Determine if corpus is primarily Chinese
        languages = [doc["language"] for doc in doc_contents]
        is_chinese = Counter(languages).most_common(1)[0][0] == "zh"

        try:
            # Create TF-IDF vectors with settings appropriate for corpus
            from core.engine.utils import configure_vectorizer
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                "zh" if is_chinese else "en",
                self.chinese_stopwords if is_chinese else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)

            # Get feature names
            feature_names = vectorizer.get_feature_names_out()

            # Check if we have enough terms
            if X.shape[1] < 5:
                return {
                    "method": "Latent Semantic Analysis",
                    "themes": [{"name": "Insufficient terms for LSA", "score": 0, "keywords": []}]
                }

            # Apply SVD to find latent semantic dimensions
            # Limit components to min(n_samples-1, n_features, 5)
            n_components = min(len(doc_texts) - 1, X.shape[1], 5)
            n_components = max(1, n_components)  # Ensure at least 1 component

            svd = TruncatedSVD(n_components=n_components)

            # Transform TF-IDF matrix to latent space
            X_svd = svd.fit_transform(X)

            # Get the most important terms for each component
            themes = []

            for i, component in enumerate(svd.components_):
                # Get top terms for this component
                # Limit to 10 terms or fewer if not enough features
                max_terms = min(10, len(feature_names))
                top_term_indices = component.argsort()[-(max_terms):][::-1]
                top_terms = [feature_names[idx] for idx in top_term_indices]

                # Filter Chinese stopwords if needed
                if is_chinese and self.chinese_stopwords:
                    top_terms = [term for term in top_terms if term not in self.chinese_stopwords]

                # Calculate explained variance for this component
                explained_variance = svd.explained_variance_ratio_[i]

                # Create theme name
                theme_name = f"Semantic Theme {i + 1}: {', '.join(top_terms[:min(3, len(top_terms))])}"

                # Find documents most associated with this component
                # Get the component scores for each document
                theme_scores = X_svd[:, i]

                # Find the top documents for this theme
                top_doc_indices = theme_scores.argsort()[::-1][:5]  # Top 5 documents
                top_doc_sources = [doc_sources[idx] for idx in top_doc_indices]

                # Get original content for these documents
                original_contents = []
                original_file_paths = []

                for source in top_doc_sources:
                    try:
                        # Construct path to original file in body directory
                        original_path = os.path.join("body", workspace, source)
                        original_file_paths.append(original_path)

                        # Read content from original file if it exists
                        if os.path.exists(original_path):
                            from core.engine.utils import get_file_content
                            content = get_file_content(original_path)
                            if content:
                                original_contents.append(content)
                            else:
                                # Fallback to the processed content
                                idx = doc_sources.index(source)
                                original_contents.append(doc_contents[idx].get("content", ""))
                        else:
                            # Fallback to the extracted content if file doesn't exist
                            idx = doc_sources.index(source)
                            original_contents.append(doc_contents[idx].get("content", ""))
                    except Exception as e:
                        debug_print(self.config, f"Error accessing original file {source}: {str(e)}")
                        # Try fallback
                        try:
                            idx = doc_sources.index(source)
                            original_contents.append(doc_contents[idx].get("content", ""))
                        except:
                            pass

                # Generate interpretive summary focusing on semantic meaning
                summary = self._generate_semantic_theme_summary(
                    original_contents,
                    top_terms,
                    top_doc_sources,
                    theme_name
                )

                themes.append({
                    "name": theme_name,
                    "score": round(float(explained_variance), 2),
                    "keywords": top_terms,
                    "documents": top_doc_sources,
                    "variance_explained": round(float(explained_variance) * 100, 1),
                    "description": summary,
                    "original_paths": original_file_paths
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

    def _output_results(self, workspace, results, method):
        """
        Output analysis results with compact colored formatting

        Args:
            workspace (str): Target workspace
            results (Dict): Analysis results
            method (str): Analysis method used
        """
        debug_print(self.config, "Outputting theme analysis results")

        # Print main header
        self.output_manager.print_formatted('header', "THEME ANALYSIS RESULTS")

        # Display results for each method
        for m, result in results.items():
            # Print method header
            self.output_manager.print_formatted('subheader', result['method'])

            # Show method-specific statistics
            stats = []
            if 'entity_count' in result:
                self.output_manager.print_formatted('kv', result['entity_count'], key="Total entities")
            if 'variance_explained' in result:
                self.output_manager.print_formatted('kv', f"{result['variance_explained']}%", key="Variance explained")
            if 'clusters' in result:
                self.output_manager.print_formatted('kv', result['clusters'], key="Number of clusters")
            if 'error' in result:
                self.output_manager.print_formatted('kv', result['error'], key="Error", success=False)

            # Display themes
            for i, theme in enumerate(result['themes']):
                if 'name' in theme:
                    # Print theme header
                    self.output_manager.print_formatted('mini_header', theme['name'])

                    # Display theme details in a compact format
                    if 'keywords' in theme:
                        self.output_manager.print_formatted('kv', ', '.join(theme['keywords']), key="Keywords",
                                                            indent=2)

                    if 'score' in theme:
                        self.output_manager.print_formatted('kv', theme['score'], key="Score", indent=2)
                    elif 'frequency' in theme:
                        self.output_manager.print_formatted('kv', theme['frequency'], key="Frequency", indent=2)
                    elif 'centrality' in theme:
                        self.output_manager.print_formatted('kv', theme['centrality'], key="Centrality", indent=2)

                    # Display document count
                    if 'document_count' in theme:
                        self.output_manager.print_formatted('kv', theme['document_count'], key="Documents", indent=2)
                    elif 'size' in theme:
                        self.output_manager.print_formatted('kv', theme['size'], key="Size", indent=2)

                    if 'variance_explained' in theme:
                        self.output_manager.print_formatted('kv', f"{theme['variance_explained']}%",
                                                            key="Variance explained", indent=2)

                    # Display detailed document list
                    if 'documents' in theme and isinstance(theme['documents'], list):
                        print("\n  Document files:")
                        for doc in theme['documents']:
                            if isinstance(doc, str):
                                self.output_manager.print_formatted('list', doc, indent=4)
                            elif isinstance(doc, dict) and 'source' in doc:
                                self.output_manager.print_formatted('list', doc['source'], indent=4)

                    # Display the LLM-generated interpretive summary
                    if 'description' in theme and theme['description']:
                        print("\n  Summary:")
                        self.output_manager.print_formatted('code', theme['description'], indent=2)

                elif 'keyword' in theme:
                    # For keyword themes - compact format
                    self.output_manager.print_formatted('mini_header', f"Keyword: {theme['keyword']}")
                    self.output_manager.print_formatted('kv', theme['score'], key="Score", indent=2)

                    # Display document count
                    if isinstance(theme['documents'], int):
                        self.output_manager.print_formatted('kv', theme['documents'], key="Documents", indent=2)

                    # Display source documents for this keyword
                    if hasattr(self, 'keyword_doc_mapping') and theme['keyword'] in self.keyword_doc_mapping:
                        doc_sources = self.keyword_doc_mapping[theme['keyword']]
                        print("\n  Document files:")
                        for source in doc_sources:
                            self.output_manager.print_formatted('list', source, indent=4)

        # Save results to file
        output_format = self.config.get('system.output_format')
        filepath = self.output_manager.save_theme_analysis(workspace, results, method, output_format)

        # Show success message
        self.output_manager.print_formatted('feedback', f"Results saved to: {filepath}")

    def _analyze_document_clusters(self, doc_contents):
        """
        Cluster documents and extract themes with access to original files in body directory

        Args:
            doc_contents (List[Dict]): Preprocessed document content

        Returns:
            Dict: Analysis results
        """
        debug_print(self.config, "Clustering documents by content similarity")

        # Extract document texts and metadata
        doc_texts = [doc["processed_content"] for doc in doc_contents]
        doc_sources = [doc["source"] for doc in doc_contents]

        # Get workspace name from config for accessing original files
        workspace = self.config.get('workspace.default')

        # Check if we have enough documents
        if len(doc_texts) < 2:
            return {
                "method": "Document Clustering",
                "themes": [{"name": "Insufficient documents for clustering", "score": 0, "documents": []}]
            }

        # Determine if corpus is primarily Chinese
        languages = [doc["language"] for doc in doc_contents]
        is_chinese = Counter(languages).most_common(1)[0][0] == "zh"

        try:
            # Create TF-IDF vectors with settings appropriate for document set size
            from core.engine.utils import configure_vectorizer
            vectorizer = configure_vectorizer(
                self.config,
                len(doc_texts),
                "zh" if is_chinese else "en",
                self.chinese_stopwords if is_chinese else None
            )

            # Transform documents to TF-IDF space
            X = vectorizer.fit_transform(doc_texts)

            # For very small document sets, limit the number of clusters
            n_docs = len(doc_texts)
            max_clusters = min(3, max(2, n_docs // 2))
            n_clusters = min(max_clusters, n_docs - 1)

            # If we only have 2 docs, force 2 clusters
            if n_docs == 2:
                n_clusters = 2

            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)

            # Extract cluster themes
            themes = []

            for cluster_id in range(n_clusters):
                # Get documents in this cluster
                cluster_doc_indices = [i for i, c in enumerate(clusters) if c == cluster_id]

                if not cluster_doc_indices:
                    continue

                cluster_docs = [doc_texts[i] for i in cluster_doc_indices]
                cluster_sources = [doc_sources[i] for i in cluster_doc_indices]

                # Access original documents from body directory
                original_contents = []
                original_file_paths = []

                for source in cluster_sources:
                    try:
                        # Construct path to original file in body directory
                        original_path = os.path.join("body", workspace, source)
                        original_file_paths.append(original_path)

                        # Read content from original file if it exists
                        if os.path.exists(original_path):
                            from core.engine.utils import get_file_content
                            content = get_file_content(original_path)
                            if content:
                                original_contents.append(content)
                            else:
                                # Fallback to the processed content if we can't read the original
                                index = cluster_doc_indices[cluster_sources.index(source)]
                                original_contents.append(doc_contents[index].get("content", ""))
                        else:
                            # Fallback to the extracted content if file doesn't exist
                            index = cluster_doc_indices[cluster_sources.index(source)]
                            original_contents.append(doc_contents[index].get("content", ""))
                    except Exception as e:
                        debug_print(self.config, f"Error accessing original file {source}: {str(e)}")
                        # Fallback to extracted content
                        index = cluster_doc_indices[cluster_sources.index(source)]
                        original_contents.append(doc_contents[index].get("content", ""))

                # Extract keywords for this cluster
                keywords = self._extract_community_keywords(cluster_docs, is_chinese)

                # Calculate cluster score using simplified approach for small document sets
                score = len(cluster_doc_indices) / len(doc_texts)

                # Create theme
                theme_name = f"Cluster: {', '.join(keywords[:min(3, len(keywords))])}"

                # Generate interpretive summary using LLM with original contents
                summary = self._generate_cluster_summary(
                    original_contents,
                    keywords,
                    cluster_sources,
                    original_file_paths
                )

                themes.append({
                    "name": theme_name,
                    "score": round(score, 2),
                    "keywords": keywords,
                    "documents": cluster_sources,
                    "document_count": len(cluster_sources),
                    "description": summary,
                    "original_paths": original_file_paths
                })

            # Sort clusters by size and score
            themes.sort(key=lambda x: (x["document_count"], x["score"]), reverse=True)

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

    def _generate_cluster_summary(self, documents, keywords, sources, original_paths=None):
        """
        Generate an interpretive summary for a document cluster using the LLM

        Args:
            documents (List[str]): Original document contents in the cluster
            keywords (List[str]): Keywords extracted for this cluster
            sources (List[str]): Document source file names
            original_paths (List[str]): Paths to original files in body directory

        Returns:
            str: Interpretive summary of the cluster
        """
        debug_print(self.config, f"Generating interpretive summary for cluster with {len(documents)} documents")

        # Make sure we have a valid LLM connector
        try:
            llm = self.factory.get_llm_connector()
            if llm is None:
                debug_print(self.config, "No LLM connector available")
                return "Summary unavailable: LLM connector not configured."
        except Exception as e:
            debug_print(self.config, f"Error getting LLM connector: {str(e)}")
            return "Summary unavailable: LLM connector error."

        # Prepare context for LLM
        try:
            # Create a prompt that uses the original documents
            prompt = f"""Summarize the common themes in the following document{'s' if len(documents) == 1 else 's'} that share these keywords: {', '.join(keywords[:8])}.

    Files: {', '.join([os.path.basename(s) for s in sources])}

    """
            # Add excerpts from each document (truncated to manage token count)
            total_excerpt_length = 0
            max_excerpt_length = 500  # Slightly longer excerpts since we have the original content
            max_total_length = 3000  # Increased total content limit

            for i, doc in enumerate(documents):
                if total_excerpt_length >= max_total_length:
                    prompt += "\n[Additional documents omitted for brevity]\n"
                    break

                file_info = f"Document {i + 1}"
                if i < len(sources):
                    file_info += f" ({os.path.basename(sources[i])})"

                # Use a decent-sized excerpt from the original content
                excerpt = doc[:max_excerpt_length] + "..." if len(doc) > max_excerpt_length else doc
                prompt += f"\n{file_info} excerpt:\n{excerpt}\n"

                total_excerpt_length += len(excerpt)

            # Add summary request with clear instructions
            if len(documents) > 1:
                prompt += "\nWrite a paragraph (5-7 sentences) summarizing the key points and themes that connect these documents. Focus on how the keywords are used across the documents and identify the main common topics. What are these documents collectively about?"
            else:
                prompt += "\nWrite a paragraph (3-5 sentences) summarizing the key points of this document that align with the identified keywords. What is this document primarily about in relation to the keywords?"

            # Use LLM to generate summary with appropriate token limit
            try:
                debug_print(self.config, "Sending summary request to LLM")
                model = self.config.get('llm.default_model')
                summary = llm.generate(prompt, model=model, max_tokens=300)

                # Check if we got a valid summary
                if summary and len(summary.strip()) > 20:
                    debug_print(self.config, "Successfully generated cluster summary")
                    return summary.strip()
                else:
                    debug_print(self.config, f"LLM returned empty or very short summary: '{summary}'")
                    return self._generate_fallback_summary(documents, keywords)

            except Exception as e:
                debug_print(self.config, f"Error in LLM.generate: {str(e)}")
                return self._generate_fallback_summary(documents, keywords)

        except Exception as e:
            debug_print(self.config, f"Error preparing summary prompt: {str(e)}")
            return self._generate_fallback_summary(documents, keywords)

    def _generate_fallback_summary(self, documents, keywords):
        """
        Generate a simple fallback summary without using LLM

        Args:
            documents (List[str]): Document contents
            keywords (List[str]): Keywords for this cluster

        Returns:
            str: Basic summary
        """
        # Create a basic summary based on document count and keywords
        if len(documents) == 1:
            return f"This document focuses on topics related to: {', '.join(keywords[:5])}. The document appears to discuss these concepts in relation to each other."
        else:
            return f"These {len(documents)} documents share themes related to: {', '.join(keywords[:5])}. They commonly discuss these concepts and their relationships, suggesting a topical connection between the documents."

    def _generate_semantic_theme_summary(self, documents, keywords, sources, theme_name):
        """
        Generate a summary focusing on the semantic meaning of a latent theme

        Args:
            documents (List[str]): Original document contents related to this theme
            keywords (List[str]): Keywords/terms from this semantic dimension
            sources (List[str]): Document source file names
            theme_name (str): Name of the semantic theme

        Returns:
            str: Interpretive summary of the semantic theme
        """
        debug_print(self.config, f"Generating semantic theme summary for {theme_name}")

        # Make sure we have a valid LLM connector
        try:
            llm = self.factory.get_llm_connector()
            if llm is None:
                debug_print(self.config, "No LLM connector available")
                return "Summary unavailable: LLM connector not configured."
        except Exception as e:
            debug_print(self.config, f"Error getting LLM connector: {str(e)}")
            return "Summary unavailable: LLM connector error."

        # Prepare context for LLM
        try:
            # Create a prompt focusing on semantic meaning
            prompt = f"""Analyze the following semantic theme extracted through Latent Semantic Analysis (LSA):

    Theme: {theme_name}

    Key terms/concepts: {', '.join(keywords)}

    This semantic dimension appears in these documents: {', '.join([os.path.basename(s) for s in sources])}

    """
            # Add excerpts from each document (truncated to manage token count)
            total_excerpt_length = 0
            max_excerpt_length = 500
            max_total_length = 3000

            for i, doc in enumerate(documents):
                if total_excerpt_length >= max_total_length:
                    prompt += "\n[Additional documents omitted for brevity]\n"
                    break

                # Use a decent-sized excerpt
                if doc:
                    excerpt = doc[:max_excerpt_length] + "..." if len(doc) > max_excerpt_length else doc
                    prompt += f"\nDocument excerpt {i + 1}:\n{excerpt}\n"
                    total_excerpt_length += len(excerpt)

            # Add summary request focused on semantic meaning
            prompt += "\nWrite a paragraph (4-6 sentences) explaining the semantic meaning of this theme. What conceptual area or topic does this combination of terms represent? How do these terms relate to each other within a coherent semantic framework? Explain what this latent theme captures about the underlying meaning in these documents."

            # Use LLM to generate summary
            try:
                debug_print(self.config, "Sending semantic theme summary request to LLM")
                model = self.config.get('llm.default_model')
                summary = llm.generate(prompt, model=model, max_tokens=300)

                # Check if we got a valid summary
                if summary and len(summary.strip()) > 20:
                    debug_print(self.config, "Successfully generated semantic theme summary")
                    return summary.strip()
                else:
                    debug_print(self.config, f"LLM returned empty or very short summary: '{summary}'")
                    return self._generate_fallback_semantic_summary(keywords, theme_name)

            except Exception as e:
                debug_print(self.config, f"Error in LLM.generate: {str(e)}")
                return self._generate_fallback_semantic_summary(keywords, theme_name)

        except Exception as e:
            debug_print(self.config, f"Error preparing semantic summary prompt: {str(e)}")
            return self._generate_fallback_semantic_summary(keywords, theme_name)

    def _generate_fallback_semantic_summary(self, keywords, theme_name):
        """
        Generate a simple fallback summary for a semantic theme without using LLM

        Args:
            keywords (List[str]): Keywords for this semantic theme
            theme_name (str): Name of the semantic theme

        Returns:
            str: Basic semantic summary
        """
        return f"This semantic theme represents a conceptual area centered around {', '.join(keywords[:3])}. These terms appear to be semantically related, suggesting a common topic or domain that connects ideas like {', '.join(keywords[3:7])}. This latent dimension captures an underlying pattern of meaning across documents that discuss these interrelated concepts."