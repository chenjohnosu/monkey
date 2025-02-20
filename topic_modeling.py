import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np
from gensim import corpora, models
from gensim.models.coherencemodel import CoherenceModel
import spacy
from llama_index.core import StorageContext, load_index_from_storage

from config import MonkeyConfig


class TopicModeler:
    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.nlp = spacy.load('en_core_web_sm')

    def _load_documents_from_vector_store(self) -> List[str]:
        """Load documents from existing vector store."""
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.config.vdb_dir
            )
            index = load_index_from_storage(storage_context)

            # Extract text from all nodes in the index
            documents = []
            for node_id, node in index.storage_context.docstore.docs.items():
                if hasattr(node, 'text'):
                    documents.append(node.text)

            return documents

        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return []

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text for topic modeling."""
        doc = self.nlp(text.lower())

        # Extract lemmatized tokens, excluding stopwords and punctuation
        tokens = [
            token.lemma_
            for token in doc
            if not token.is_stop
               and not token.is_punct
               and not token.is_space
               and len(token.text) > 2  # Filter out very short tokens
        ]

        return tokens

    def _find_optimal_topics(
            self,
            corpus: List[List[int]],
            dictionary: corpora.Dictionary,
            texts: List[List[str]],
            start: int = 2,
            limit: int = 15,
            step: int = 1
    ) -> Tuple[int, float]:
        """Find optimal number of topics using coherence scores."""
        coherence_scores = []
        models_list = []

        for num_topics in range(start, limit, step):
            lda_model = models.LdaModel(
                corpus=corpus,
                num_topics=num_topics,
                id2word=dictionary,
                random_state=42,
                passes=10
            )

            coherence_model = CoherenceModel(
                model=lda_model,
                texts=texts,
                dictionary=dictionary,
                coherence='c_v'
            )

            coherence_score = coherence_model.get_coherence()
            coherence_scores.append(coherence_score)
            models_list.append(lda_model)

            self.logger.info(f"Num Topics: {num_topics}, Coherence Score: {coherence_score}")

        optimal_num_topics = start + coherence_scores.index(max(coherence_scores))
        return optimal_num_topics, models_list[coherence_scores.index(max(coherence_scores))]

    def analyze_topics(self, num_words: int = 10) -> Dict[str, Any]:
        """Perform automatic topic analysis on documents in vector store."""
        self.logger.info("Starting topic analysis...")

        # Load documents from vector store
        documents = self._load_documents_from_vector_store()
        if not documents:
            return {"error": "No documents found in vector store"}

        # Preprocess documents
        processed_docs = [self._preprocess_text(doc) for doc in documents]

        # Create dictionary and corpus
        dictionary = corpora.Dictionary(processed_docs)
        corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

        # Find optimal number of topics
        try:
            optimal_num_topics, lda_model = self._find_optimal_topics(
                corpus,
                dictionary,
                processed_docs
            )
        except Exception as e:
            self.logger.error(f"Error finding optimal topics: {str(e)}")
            return {"error": f"Topic modeling failed: {str(e)}"}

        # Extract topics
        topics = {}
        for topic_id in range(optimal_num_topics):
            word_probs = lda_model.show_topic(topic_id, num_words)
            topics[f"Topic {topic_id + 1}"] = {
                "words": [word for word, _ in word_probs],
                "probabilities": [float(prob) for _, prob in word_probs],
                "coherence": float(
                    CoherenceModel(
                        model=lda_model,
                        texts=processed_docs,
                        dictionary=dictionary,
                        coherence='c_v'
                    ).get_coherence_per_topic()[topic_id]
                )
            }

        # Document-topic distribution
        doc_topics = []
        for doc, orig_doc in zip(corpus, documents):
            topic_dist = lda_model.get_document_topics(doc)
            main_topic = max(topic_dist, key=lambda x: x[1])
            doc_topics.append({
                "main_topic": f"Topic {main_topic[0] + 1}",
                "probability": float(main_topic[1]),
                "doc_preview": orig_doc[:200] + "..."  # First 200 chars as preview
            })

        return {
            "num_topics": optimal_num_topics,
            "topics": topics,
            "document_topics": doc_topics,
            "num_documents": len(documents)
        }

    def print_analysis(self, analysis: Dict[str, Any]) -> None:
        """Print topic analysis results in a readable format."""
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return

        print("\n=== Topic Analysis Results ===")
        print(f"Number of documents analyzed: {analysis['num_documents']}")
        print(f"Optimal number of topics: {analysis['num_topics']}")

        print("\n=== Topics ===")
        for topic_name, topic_data in analysis['topics'].items():
            print(f"\n{topic_name}")
            print("Words:", ", ".join(topic_data['words']))
            print(f"Coherence: {topic_data['coherence']:.3f}")

        print("\n=== Document Classifications ===")
        for i, doc in enumerate(analysis['document_topics'], 1):
            print(f"\nDocument {i}:")
            print(f"Main Topic: {doc['main_topic']}")
            print(f"Confidence: {doc['probability']:.2%}")
            print(f"Preview: {doc['doc_preview']}")