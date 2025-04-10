"""
Topic modeling module for extracting main topics from document collections
"""

import os
import re
import numpy as np
from collections import Counter, defaultdict
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.engine.utils import ensure_dir
from core.engine.logging import debug_print,warning,info
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.language.tokenizer import ChineseTokenizer, JIEBA_AVAILABLE

try:
    from sklearn.decomposition import LatentDirichletAllocation, NMF
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warning("scikit-learn not available, falling back to simplified topic modeling")

try:
    import umap
    import hdbscan
    CLUSTERING_AVAILABLE = True
except ImportError:
    CLUSTERING_AVAILABLE = False
    warning("umap and/or hdbscan not available, some clustering features will be limited")

try:
    import jieba
    JIEBA_AVAILABLE = True
except ImportError:
    JIEBA_AVAILABLE = False
    warning("jieba not available, falling back to character-based segmentation for Chinese")

class TopicModeler:

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the topic modeler"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)

        # Initialize LLM connector for topic interpretation
        try:
            from core.connectors.connector_factory import ConnectorFactory
            self.factory = ConnectorFactory(config)
            self.llm_connector = self.factory.get_llm_connector()
            debug_print(config, "LLM connector initialized for topic modeling")
        except Exception as e:
            debug_print(config, f"Error initializing LLM connector: {str(e)}")
            self.factory = None
            self.llm_connector = None

        # Load stopwords
        self.chinese_stopwords = self._load_stopwords('chinese')
        self.english_stopwords = self._load_stopwords('english')

        debug_print(config, "Topic modeler initialized")

    def _load_stopwords(self, language):
        """
        Load stopwords for a given language

        Args:
            language (str): Language to load stopwords for

        Returns:
            set: Stopwords for the specified language
        """
        stopwords = set()
        lexicon_dir = 'lexicon'
        ensure_dir(lexicon_dir)

        filename = f'stopwords_{language}.txt'
        filepath = os.path.join(lexicon_dir, filename)

        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                stopwords = set(line.strip() for line in file if line.strip())
            debug_print(self.config, f"Loaded {len(stopwords)} {language} stopwords")
        except FileNotFoundError:
            debug_print(self.config, f"No {language} stopwords file found")

            # Fallback to default stopwords
            if language == 'english':
                stopwords = {
                    "the", "and", "a", "to", "of", "in", "is", "that", "it",
                    "with", "for", "as", "was", "on", "are", "by", "this"
                }
            elif language == 'chinese':
                stopwords = {"的", "了", "和", "是", "就", "都", "而", "及"}

        return stopwords

    def analyze(self, workspace, method='all'):
        """
        Analyze topics in a workspace

        Args:
            workspace (str): The workspace to analyze
            method (str): Analysis method ('all', 'lda', 'nmf', 'cluster')

        Returns:
            Dict: Topic analysis results
        """
        debug_print(self.config, f"Analyzing topics in workspace '{workspace}' using method '{method}'")

        # Validate method - case insensitive matching
        valid_methods = ['all', 'lda', 'nmf', 'cluster']
        lower_method = method.lower() if isinstance(method, str) else ''

        if lower_method not in valid_methods:
            print(f"Invalid method: {method}. Must be one of: {', '.join(valid_methods)}")
            return {}

        # Convert to lowercase for consistent handling
        method = lower_method

        # Validate workspace and load documents
        docs = self._validate_and_load_documents(workspace)
        if not docs:
            print(f"ERROR: No documents found in workspace '{workspace}'")
            return {
                "method": "Topic Modeling",
                "error": "No documents found in workspace",
                "topics": []
            }

        # Preprocess documents
        try:
            doc_contents, language_counts = self._preprocess_documents(docs)
        except Exception as e:
            print(f"ERROR: Failed to preprocess documents: {str(e)}")
            return {
                "method": "Topic Modeling",
                "error": f"Preprocessing failed: {str(e)}",
                "topics": []
            }

        # Display language distribution
        self._display_language_stats(language_counts)

        # Print preprocessing details for debugging
        print(f"Preprocessed documents: {len(doc_contents)}")
        if doc_contents:
            print("Sample document:")
            sample_doc = doc_contents[0]
            print(f"  Source: {sample_doc.get('source', 'Unknown')}")
            print(f"  Language: {sample_doc.get('language', 'Unknown')}")
            print(f"  Content preview: {sample_doc.get('content', '')[:200]}...")

        # Run topic modeling
        try:
            results = self._run_topic_modeling(doc_contents, method)
        except Exception as e:
            print(f"ERROR: Topic modeling failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                "method": "Topic Modeling",
                "error": f"Topic modeling failed: {str(e)}",
                "topics": []
            }

        # Verify results
        if not results:
            print("WARNING: No topic modeling results generated")
            return {
                "method": "Topic Modeling",
                "error": "No topics could be generated",
                "topics": []
            }

        # Output and save results
        return self._output_results(workspace, results, method)

    def _validate_and_load_documents(self, workspace):
        """
        Validate workspace and load documents for topic modeling

        Args:
            workspace (str): The workspace to load documents from

        Returns:
            List[Dict]: Documents in the workspace
        """
        debug_print(self.config, f"Validating and loading documents for workspace: {workspace}")

        # Check if workspace exists
        data_dir = os.path.join("data", workspace)
        if not os.path.exists(data_dir):
            print(f"Workspace '{workspace}' does not exist")
            return []

        # Load documents from storage
        docs = self.storage_manager.get_documents(workspace)

        # Print initial document loading details
        print(f"\nDocument Loading:")
        print(f"  Workspace: {workspace}")
        print(f"  Total documents found: {len(docs)}")

        if not docs:
            print("  WARNING: No documents found in workspace")

        # Detailed document info logging
        if docs:
            print("  Sample Document Details:")
            sample_doc = docs[0]
            source = sample_doc.get('metadata', {}).get('source', 'Unknown')
            language = sample_doc.get('metadata', {}).get('language', 'unknown')
            content_length = len(sample_doc.get('content', ''))
            processed_content_length = len(sample_doc.get('processed_content', ''))

            print(f"    Source: {source}")
            print(f"    Language: {language}")
            print(f"    Raw Content Length: {content_length}")
            print(f"    Processed Content Length: {processed_content_length}")

        return docs

    def _preprocess_documents(self, docs):
        """
        Preprocess documents for topic modeling, incorporating original files from body directory

        Args:
            docs (List[Dict]): Raw input documents from data directory

        Returns:
            Tuple[List[Dict], Counter]: Processed documents and language counts
        """
        processed_docs = []
        language_counts = Counter()

        # Get current workspace
        workspace = self.config.get('workspace.default')

        for doc in docs:
            # Extract content and metadata
            content = doc.get("content", "").strip()
            processed_content = doc.get("processed_content", "").strip()
            source_path = doc.get("metadata", {}).get("source", "unknown")
            language = doc.get("metadata", {}).get("language", "unknown")

            # Try to get original content from body directory if needed
            if not content or self.config.get('topic.use_originals', True):
                try:
                    # Construct path to original file in body directory
                    original_path = os.path.join("body", workspace, source_path)

                    if os.path.exists(original_path):
                        debug_print(self.config, f"Loading original content from: {original_path}")
                        from core.engine.utils import get_file_content
                        original_content = get_file_content(original_path)

                        if original_content:
                            # Use original content if available
                            content = original_content
                            debug_print(self.config, f"Using original content for: {source_path}")
                        else:
                            debug_print(self.config, f"Failed to load original content, using stored content")
                    else:
                        debug_print(self.config, f"Original file not found: {original_path}")
                except Exception as e:
                    debug_print(self.config, f"Error loading original file: {str(e)}")

            # Prefer processed content for better topic modeling results
            text_to_use = processed_content if processed_content else content

            # Skip empty documents
            if not text_to_use:
                debug_print(self.config, f"Empty document skipped: {source_path}")
                continue

            # Track language
            language_counts[language] += 1

            processed_docs.append({
                "content": text_to_use,
                "source": source_path,
                "language": language,
                "original_content": content  # Store original content for reference
            })

        print(f"\nPreprocessing Summary:")
        print(f"  Total Input Documents: {len(docs)}")
        print(f"  Processed Documents: {len(processed_docs)}")
        if processed_docs:
            print("  Sample Processed Document:")
            sample_doc = processed_docs[0]
            print(f"    Source: {sample_doc['source']}")
            print(f"    Language: {sample_doc['language']}")
            print(f"    Content Preview: {sample_doc['content'][:200]}...")

        if not processed_docs:
            print("ERROR: No documents could be preprocessed")

        return processed_docs, language_counts

    def _display_language_stats(self, language_counts):
        """
        Display language distribution of documents

        Args:
            language_counts (Counter): Language distribution
        """
        print("\nDocument Language Distribution:")
        total = sum(language_counts.values())
        for lang, count in language_counts.most_common():
            percentage = (count / total) * 100
            print(f"  {lang}: {count} documents ({percentage:.1f}%)")

    def _run_topic_modeling(self, docs, method):
        """
        Run topic modeling based on selected method

        Args:
            docs (List[Dict]): Preprocessed documents
            method (str): Topic modeling method

        Returns:
            Dict: Topic modeling results
        """
        # Group documents by language
        docs_by_language = defaultdict(list)
        for doc in docs:
            docs_by_language[doc['language']].append(doc)

        results = {}

        # Check library availability
        if not SKLEARN_AVAILABLE:
            print("scikit-learn not available. Using simple extraction.")
            method = 'simple'

        # Log which method we're executing
        print(f"\nExecuting topic modeling method: {method}")

        # Run topic modeling for each language
        for language, language_docs in docs_by_language.items():
            # Skip if too few documents
            if len(language_docs) < 3:
                continue

            # Run specific methods based on what was requested
            if method == 'all' or method == 'lda':
                print(f"Running LDA topic modeling for language: {language}")
                results[f'lda_{language}'] = self._lda_topic_modeling(language_docs, language)

            if method == 'all' or method == 'nmf':
                print(f"Running NMF topic modeling for language: {language}")
                results[f'nmf_{language}'] = self._nmf_topic_modeling(language_docs, language)

            if (method == 'all' or method == 'cluster') and CLUSTERING_AVAILABLE:
                print(f"Running clustering-based topic modeling for language: {language}")
                results[f'cluster_{language}'] = self._cluster_topic_modeling(language_docs, language)

        return results

    def _process_and_save_results(self, workspace, results, method):
        """
        Process and save topic modeling results

        Args:
            workspace (str): Workspace name
            results (Dict): Topic modeling results
            method (str): Analysis method

        Returns:
            Dict: Processed results
        """
        debug_print(self.config, "Processing and saving topic modeling results")
        processed_results = {}

        # Handle empty results or non-dictionary results
        if not results:
            print("No topic modeling results to process.")
            return {}

        # Process each result set
        for key, result in results.items():
            try:
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    processed_result = {
                        "method": "Unknown Topic Modeling",
                        "error": f"Invalid result format for {key}",
                        "topics": []
                    }
                else:
                    # Use the existing result or create a default structure
                    processed_result = result.copy()

                    # Derive method if not present
                    if 'method' not in processed_result:
                        if 'lda' in key:
                            processed_result['method'] = 'Latent Dirichlet Allocation'
                        elif 'nmf' in key:
                            processed_result['method'] = 'Non-Negative Matrix Factorization'
                        elif 'cluster' in key:
                            processed_result['method'] = 'Clustering-based Topic Modeling'
                        else:
                            processed_result['method'] = 'Topic Modeling'

                    # Ensure topics exist
                    if 'topics' not in processed_result:
                        processed_result['topics'] = []

                    # Add language if not present
                    if 'language' not in processed_result:
                        language_match = re.search(r'_(en|zh)$', key)
                        processed_result['language'] = language_match.group(1) if language_match else 'unknown'

                processed_results[key] = processed_result

            except Exception as e:
                debug_print(self.config, f"Error processing result for {key}: {str(e)}")
                processed_results[key] = {
                    "method": "Error Processing Result",
                    "error": str(e),
                    "topics": []
                }

        # Prepare data for saving
        data = {
            'timestamp': datetime.now().isoformat(),
            'workspace': workspace,
            'method': method,
            'results': processed_results
        }

        # Save results using output manager
        output_format = self.config.get('system.output_format', 'txt')
        filepath = self.output_manager.save_topic_analysis(workspace, data, method, output_format)

        # Print results summary
        print(f"\nTopic Analysis Results saved to: {filepath}")

        return processed_results

    def _lda_topic_modeling(self, docs, language):
        """
        Perform Latent Dirichlet Allocation topic modeling

        Args:
            docs (List[Dict]): Documents to analyze
            language (str): Language of documents

        Returns:
            Dict: LDA topic analysis results
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}

        try:
            # Prepare documents
            doc_texts = [doc["content"] for doc in docs]

            # Configure vectorization
            vectorizer = TfidfVectorizer(
                stop_words=list(self.chinese_stopwords) if language == 'zh' else 'english',
                tokenizer=ChineseTokenizer() if language == 'zh' else None,
                max_df=0.95,
                min_df=2
            )

            # Create document-term matrix
            doc_term_matrix = vectorizer.fit_transform(doc_texts)

            # Determine number of topics
            n_topics = min(max(5, len(docs) // 3), 20)

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
                top_docs = [docs[i]["source"] for i in top_docs_idx]

                topics.append({
                    "name": f"Topic {topic_idx + 1}",
                    "keywords": top_words,
                    "documents": top_docs,
                    "score": topic.max() / topic.sum()
                })

            return {
                "method": "Latent Dirichlet Allocation",
                "language": language,
                "topics": topics
            }

        except Exception as e:
            debug_print(self.config, f"LDA Topic Modeling Error: {str(e)}")
            return {"error": str(e)}

    def _nmf_topic_modeling(self, docs, language):
        """
        Perform Non-Negative Matrix Factorization topic modeling with version compatibility

        Args:
            docs (List[Dict]): Documents to analyze
            language (str): Language of documents

        Returns:
            Dict: NMF topic analysis results
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available"}

        try:
            # Prepare documents
            doc_texts = [doc["content"] for doc in docs]

            # Configure vectorization
            vectorizer = TfidfVectorizer(
                stop_words=list(self.chinese_stopwords) if language == 'zh' else 'english',
                tokenizer=ChineseTokenizer() if language == 'zh' else None,
                max_df=0.95,  # Ignore terms that appear in more than 95% of documents
                min_df=2,  # Ignore terms that appear in fewer than 2 documents
                smooth_idf=True  # Smooth idf weights to prevent divide-by-zero
            )

            # Create document-term matrix
            doc_term_matrix = vectorizer.fit_transform(doc_texts)

            # Determine number of topics dynamically
            n_topics = min(max(5, len(docs) // 3), 20)

            # Try to detect scikit-learn version and use compatible parameters
            from sklearn import __version__ as sklearn_version
            debug_print(self.config, f"scikit-learn version: {sklearn_version}")

            # Initialize NMF with minimal parameters to ensure compatibility
            try:
                # First try with just the essential parameters
                nmf_model = NMF(
                    n_components=n_topics,
                    random_state=42,
                    max_iter=200
                )
                debug_print(self.config, "Using basic NMF parameters")
            except Exception as e1:
                debug_print(self.config, f"Basic NMF initialization failed: {str(e1)}")
                # If that fails, try with absolutely minimal parameters
                try:
                    nmf_model = NMF(n_components=n_topics)
                    debug_print(self.config, "Using minimal NMF parameters")
                except Exception as e2:
                    debug_print(self.config, f"Minimal NMF initialization failed: {str(e2)}")
                    # If even that fails, return an error
                    return {"error": f"Could not initialize NMF model: {str(e2)}"}

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
                top_docs = [docs[i]["source"] for i in top_docs_idx]

                # Generate topic summary using LLM if available
                summary = None
                if self.llm_connector:
                    try:
                        prompt = f"""Analyze the following topic keywords and identify the core theme.
                        What conceptual area do these words represent?
                        Explain the interconnections between these keywords and their significance.

                        Keywords: {', '.join(top_words)}"""

                        model = self.config.get('llm.default_model', 'mistral')
                        summary = self.llm_connector.generate(
                            prompt,
                            model=model,
                            max_tokens=300
                        )
                    except Exception as e:
                        debug_print(self.config, f"Error generating topic summary: {str(e)}")

                # Create topic representation
                topics.append({
                    "name": f"NMF Topic {topic_idx + 1}",
                    "keywords": top_words,
                    "documents": top_docs,
                    "score": topic.max() / topic.sum(),
                    "summary": summary,
                    "language": language
                })

            # Compute overall topic coherence
            try:
                # Simple coherence measure: average uniqueness of top words
                word_frequency = Counter(
                    word for topic in topics for word in topic['keywords']
                )

                # Adjust topic scores based on word uniqueness
                for topic in topics:
                    uniqueness_score = sum(
                        1 / word_frequency[word] for word in topic['keywords']
                    ) / len(topic['keywords'])
                    topic['coherence'] = uniqueness_score
            except Exception as e:
                debug_print(self.config, f"Error computing topic coherence: {str(e)}")

            return {
                "method": "Non-Negative Matrix Factorization",
                "language": language,
                "topics": topics,
                "num_topics": n_topics
            }

        except Exception as e:
            debug_print(self.config, f"NMF Topic Modeling Error: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return {"error": str(e)}

    def _cluster_topic_modeling(self, docs, language):
        """
        Perform Clustering-based topic modeling

        Args:
            docs (List[Dict]): Documents to analyze
            language (str): Language of documents

        Returns:
            Dict: Clustering topic analysis results
        """
        if not SKLEARN_AVAILABLE:
            return {"error": "scikit-learn not available for clustering"}

        try:
            # Prepare documents
            doc_texts = [doc["content"] for doc in docs]
            doc_sources = [doc["source"] for doc in docs]

            # Configure vectorization
            vectorizer = TfidfVectorizer(
                stop_words=list(self.chinese_stopwords) if language == 'zh' else 'english',
                tokenizer=ChineseTokenizer() if language == 'zh' else None,
                max_df=0.95,
                min_df=2
            )

            # Create document-term matrix
            X = vectorizer.fit_transform(doc_texts)

            # Determine number of clusters dynamically
            n_docs = len(doc_texts)

            # Define number of clusters based on document count
            if n_docs <= 5:
                n_clusters = 2
            elif n_docs <= 10:
                n_clusters = 3
            elif n_docs <= 20:
                n_clusters = 4
            else:
                n_clusters = min(int(n_docs / 5), 10)  # 1 cluster per 5 docs, max 10

            # Perform clustering using K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X)

            # Extract feature names
            feature_names = vectorizer.get_feature_names_out()

            # Process clusters
            topics = []
            for cluster_id in range(n_clusters):
                # Get documents in this cluster
                cluster_mask = (cluster_labels == cluster_id)
                cluster_docs = [doc_texts[i] for i in range(len(doc_texts)) if cluster_mask[i]]
                cluster_sources = [doc_sources[i] for i in range(len(doc_texts)) if cluster_mask[i]]

                # Extract keywords for this cluster
                keywords = self._extract_cluster_keywords(cluster_docs, language)

                # Calculate top words for the cluster
                cluster_vector = kmeans.cluster_centers_[cluster_id]
                top_word_indices = cluster_vector.argsort()[-10:][::-1]
                top_words = [feature_names[i] for i in top_word_indices]

                # Create topic representation
                topic = {
                    "name": f"Cluster {cluster_id + 1}",
                    "keywords": keywords,
                    "documents": cluster_sources,
                    "document_count": len(cluster_docs),
                    "score": cluster_vector.max() / cluster_vector.sum(),
                    "top_words": top_words
                }

                topics.append(topic)

            # Sort topics by document count and score
            topics.sort(key=lambda x: (x['document_count'], x['score']), reverse=True)

            return {
                "method": "Document Clustering",
                "language": language,
                "clusters": n_clusters,
                "topics": topics
            }

        except Exception as e:
            debug_print(self.config, f"Clustering Topic Modeling Error: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return {"error": str(e)}

    def _extract_cluster_keywords(self, docs, language):
        """
        Extract keywords for a cluster of documents

        Args:
            docs (List[str]): Documents in the cluster
            language (str): Language of documents

        Returns:
            List[str]: Top keywords for the cluster
        """
        try:
            # Combine all documents
            combined_text = " ".join(docs)

            # Vectorize and extract top keywords
            vectorizer = TfidfVectorizer(
                stop_words=list(self.chinese_stopwords) if language == 'zh' else 'english',
                tokenizer=ChineseTokenizer() if language == 'zh' else None,
                max_df=0.95,
                min_df=1
            )

            # Create a small corpus with just the combined text
            X = vectorizer.fit_transform([combined_text])
            feature_names = vectorizer.get_feature_names_out()

            # Get TF-IDF scores
            scores = X.toarray()[0]

            # Sort words by score
            scored_terms = sorted(
                zip(feature_names, scores),
                key=lambda x: x[1],
                reverse=True
            )

            # Return top 10 keywords
            return [term for term, score in scored_terms[:10]]

        except Exception as e:
            debug_print(self.config, f"Error extracting cluster keywords: {str(e)}")
            return []

    def _output_results(self, workspace, results, method):
        """
        Process and output topic modeling results

        Args:
            workspace (str): Workspace name
            results (Dict): Topic modeling results
            method (str): Analysis method

        Returns:
            Dict: Processed results
        """
        debug_print(self.config, "Processing and saving topic modeling results")

        # Print main header
        self.output_manager.print_formatted('header', "TOPIC MODELING RESULTS")

        # Handle empty results or non-dictionary results
        if not results:
            print("No topic modeling results to process.")
            return {}

        # Process each result set
        for key, result in results.items():
            try:
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    print(f"Invalid result format for {key}")
                    continue

                # Display method and language
                method_name = result.get('method', 'Unknown Topic Modeling')
                language = result.get('language', 'unknown')

                # Print method header
                self.output_manager.print_formatted('subheader', f"{method_name} ({language.upper()})")

                # Handle potential error in results
                if 'error' in result:
                    self.output_manager.print_formatted('feedback', f"Error: {result['error']}", success=False)
                    continue

                # Display topics
                topics = result.get('topics', [])
                print(f"Total Topics: {len(topics)}")

                # Display each topic
                for i, topic in enumerate(topics, 1):
                    # Print topic name
                    self.output_manager.print_formatted('mini_header', topic.get('name', f'Topic {i}'))

                    # Keywords
                    keywords = topic.get('keywords', [])
                    if keywords:
                        self.output_manager.print_formatted('kv', ', '.join(keywords), key="Keywords")

                    # Score or Coherence
                    if 'score' in topic:
                        self.output_manager.print_formatted('kv', f"{topic['score']:.4f}", key="Score")
                    elif 'coherence' in topic:
                        self.output_manager.print_formatted('kv', f"{topic['coherence']:.4f}", key="Coherence")

                    # Documents
                    documents = topic.get('documents', [])
                    if documents:
                        self.output_manager.print_formatted('kv', len(documents), key="Documents")

                        # Show first few document sources
                        print("  Document Sources:")
                        for doc in documents[:5]:
                            self.output_manager.print_formatted('list', doc, indent=4)

                        if len(documents) > 5:
                            print(f"  ... and {len(documents) - 5} more")

                    # Summary (if available)
                    summary = topic.get('summary')
                    if summary:
                        print("\n  Summary:")
                        self.output_manager.print_formatted('code', summary, indent=2)

            except Exception as e:
                debug_print(self.config, f"Error processing result for {key}: {str(e)}")
                print(f"Error processing result: {str(e)}")

        # Prepare data for saving
        output_format = self.config.get('system.output_format', 'txt')
        filepath = self.output_manager.save_topic_analysis(workspace, results, method, output_format)

        # Print save location
        self.output_manager.print_formatted('feedback', f"Results saved to: {filepath}")

        return results