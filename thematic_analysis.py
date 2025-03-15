import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from collections import Counter
import torch
import spacy
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from gensim import corpora
from llama_index.core import StorageContext, load_index_from_storage

from config import MonkeyConfig


class ThematicAnalyzer:
    """Performs comprehensive thematic analysis across all documents."""

    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Try to load spacy model, download if needed
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            print("Downloading spaCy model (this will only happen once)...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')

        print(f"Initialized ThematicAnalyzer with config: vdb_dir={config.vdb_dir}")

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
                        'metadata': {}
                    }

                    # Extract metadata if available
                    if hasattr(node, 'metadata'):
                        doc_info['metadata'] = node.metadata

                    documents.append(doc_info)

            print(f"Loaded {len(documents)} document chunks from vector store")
            return documents

        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            print(f"Error loading vector store: {str(e)}")
            return []

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for analysis, extracting key terms."""
        doc = self.nlp(text.lower())

        # Extract lemmatized tokens, focusing on nouns, verbs, adjectives
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
               and not token.is_punct
               and not token.is_space
               and len(token.text) > 2
               and token.pos_ in ('NOUN', 'VERB', 'ADJ', 'PROPN')
        ]

        return tokens

    def _extract_key_phrases(self, text: str, top_n: int = 20) -> List[str]:
        """Extract meaningful key phrases from text."""
        doc = self.nlp(text)

        # Get noun chunks (noun phrases)
        chunks = [chunk.text.lower() for chunk in doc.noun_chunks
                  if len(chunk.text.split()) > 1]  # Only multi-word phrases

        # Get named entities
        entities = [ent.text.lower() for ent in doc.ents]

        # Combine and count occurrences
        all_phrases = chunks + entities
        phrase_counts = Counter(all_phrases)

        # Return most common phrases
        return [phrase for phrase, _ in phrase_counts.most_common(top_n)]

    def identify_themes_with_nmf(self, n_themes: int = 10) -> Dict[str, Any]:
        """Identify themes using Non-Negative Matrix Factorization (NMF)."""
        documents = self._load_documents_from_vector_store()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Extract text content
        doc_texts = [doc['text'] for doc in documents]
        doc_metadata = [doc.get('metadata', {}) for doc in documents]

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
            stop_words='english'
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
                    'score': float(score)
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
            'themes': themes
        }

    def identify_concept_network(self, min_co_occurrence: int = 3) -> Dict[str, Any]:
        """Build a network of co-occurring concepts across documents."""
        documents = self._load_documents_from_vector_store()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Process documents to extract key terms
        doc_terms = []
        for doc in documents:
            terms = self._preprocess_text(doc['text'])
            doc_terms.append(terms)

        # Build term co-occurrence matrix
        term_counter = Counter()
        for terms in doc_terms:
            term_counter.update(terms)

        # Filter to keep only common terms
        common_terms = [term for term, count in term_counter.items()
                        if count >= min_co_occurrence]

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

            return {
                'method': 'Concept Co-occurrence Network',
                'num_documents': len(documents),
                'num_concepts': len(common_terms),
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

            return {
                'method': 'Concept Co-occurrence Network (without community detection)',
                'num_documents': len(documents),
                'num_concepts': len(common_terms),
                'top_concepts': [{"concept": node, "centrality": float(score)}
                                 for node, score in top_concepts],
                'network_stats': {
                    'nodes': G.number_of_nodes(),
                    'edges': G.number_of_edges()
                }
            }

    def extract_common_keyphrases(self, min_doc_frequency: int = 2) -> Dict[str, Any]:
        """Extract common key phrases across all documents."""
        documents = self._load_documents_from_vector_store()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Extract phrases from each document
        doc_phrases = []
        for doc in documents:
            phrases = self._extract_key_phrases(doc['text'])
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

        return {
            'method': 'Common Key Phrases',
            'num_documents': doc_count,
            'common_phrases': [
                {
                    'phrase': phrase,
                    'doc_count': phrase_doc_count[phrase],
                    'coverage_percentage': float(coverage)
                }
                for phrase, coverage in sorted_phrases[:30]  # Top 30 phrases
            ]
        }

    def analyze_all_themes(self) -> Dict[str, Any]:
        """Perform comprehensive thematic analysis using multiple methods."""
        results = {}

        print("Starting comprehensive thematic analysis...")
        overall_start = time.time()

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

    def print_analysis_results(self, results: Dict[str, Any]) -> None:
        """Print thematic analysis results in a formatted way."""
        print("\n===== COMPREHENSIVE THEMATIC ANALYSIS =====\n")

        # Print NMF themes
        if 'nmf_themes' in results and 'error' not in results['nmf_themes']:
            nmf = results['nmf_themes']
            print(f"\n--- Thematic Topics ({nmf['num_themes']}) ---")

            for theme_name, theme_data in nmf['themes'].items():
                print(f"\n{theme_name}")
                print(f"Key terms: {', '.join(theme_data['terms'][:5])}")
                print(f"Prevalence: {theme_data['prevalence']:.1%}")
                print("Top documents:")
                for doc in theme_data['top_documents'][:3]:
                    print(f"  - {doc['name']} (relevance: {doc['score']:.2f})")

        # Print concept network results
        if 'concept_network' in results and 'error' not in results['concept_network']:
            network = results['concept_network']
            print(f"\n\n--- Concept Network Analysis ---")
            print(
                f"Identified {network['network_stats']['nodes']} key concepts with {network['network_stats']['edges']} connections")

            print("\nTop Concepts by Centrality:")
            for concept in network['top_concepts'][:10]:
                print(f"  - {concept['concept']} (centrality: {concept['centrality']:.3f})")

            if 'concept_groups' in network:
                print("\nConcept Groups:")
                for group_name, concepts in list(network['concept_groups'].items())[:5]:  # Show top 5 groups
                    print(f"  {group_name}: {', '.join(concepts[:5])}" +
                          (f" and {len(concepts) - 5} more" if len(concepts) > 5 else ""))

        # Print key phrases
        if 'key_phrases' in results and 'error' not in results['key_phrases']:
            phrases = results['key_phrases']
            print(f"\n\n--- Common Key Phrases ---")
            print(f"Across {phrases['num_documents']} documents")

            print("\nMost Common Phrases:")
            for phrase_data in phrases['common_phrases'][:15]:
                print(
                    f"  - '{phrase_data['phrase']}' (in {phrase_data['doc_count']} docs, {phrase_data['coverage_percentage']:.1f}%)")