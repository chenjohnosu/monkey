"""
Haystack integration for pipeline-based document processing
"""

import os
from typing import List, Dict, Any, Optional
from core.engine.utils import ensure_dir
from core.engine.logging import debug

class HaystackConnector:
    """Provides integration with Haystack for pipeline-based document processing"""
    
    def __init__(self, config):
        """Initialize Haystack connector with configuration"""
        self.config = config
        self.pipelines = {}
        debug(config, "Haystack connector initialized")
        
    def init_document_store(self, workspace: str) -> bool:
        """
        Initialize a document store for a workspace
        
        Args:
            workspace (str): Target workspace
            
        Returns:
            bool: Success flag
        """
        debug(self.config, f"Initializing Haystack document store for workspace: {workspace}")
        
        try:
            # Import Haystack components
            from haystack.document_stores import InMemoryDocumentStore
            from haystack.components.embedders import SentenceTransformersDocumentEmbedder
            from haystack.components.retrievers import InMemoryEmbeddingRetriever
            
            # Get document store directory
            store_dir = os.path.join("data", workspace, "document_store")
            ensure_dir(store_dir)
            
            # Create document store
            self.document_store = InMemoryDocumentStore()
            
            # Get embedding model from config
            embedding_model_name = self.config.get('embedding.default_model')
            
            # Configure embedding model
            if embedding_model_name == "multilingual-e5":
                model_name = "intfloat/multilingual-e5-large"
            elif embedding_model_name == "mixbread":
                model_name = "mixedbread-ai/mxbai-embed-large-v1"
            elif embedding_model_name == "jina-zh":
                model_name = "jinaai/jina-embeddings-v2-base-zh"
            else:
                # Default to a good multilingual model
                model_name = "intfloat/multilingual-e5-large"
            
            # Create the embedder and retriever components
            self.embedder = SentenceTransformersDocumentEmbedder(model_name_or_path=model_name)
            
            # Create retriever
            self.retriever = InMemoryEmbeddingRetriever(
                document_store=self.document_store,
                embedding_similarity_function="cosine"
            )
            
            return True
            
        except Exception as e:
            debug(self.config, f"Error initializing Haystack document store: {str(e)}")
            print(f"Error initializing document store: {str(e)}")
            return False

    def add_documents(self, workspace: str, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the document store

        Args:
            workspace (str): Target workspace
            documents (List[Dict]): List of document objects with content and metadata

        Returns:
            bool: Success flag
        """
        debug(self.config, f"Adding {len(documents)} documents to Haystack document store")

        # Initialize document store if needed
        if not hasattr(self, 'document_store'):
            if not self.init_document_store(workspace):
                return False

        try:
            # Import Haystack document model
            from haystack.schema import Document as HaystackDocument

            # Convert to Haystack documents, explicitly using processed content
            haystack_docs = []
            for doc in documents:
                # Extract content and metadata - EXPLICITLY USE PROCESSED CONTENT WITH FALLBACK
                content = doc.get("processed_content", doc.get("content", ""))
                metadata = doc.get("metadata", {}).copy()

                # Create Haystack document with processed content
                haystack_doc = HaystackDocument(
                    content=content,  # Using processed content with stop words removed
                    meta=metadata
                )
                haystack_docs.append(haystack_doc)

            # Run the embedding pipeline on the documents
            embedded_docs = self.embedder.run(documents=haystack_docs)["documents"]

            # Write documents to the document store
            self.document_store.write_documents(embedded_docs)

            # Save document store metadata
            store_dir = os.path.join("data", workspace, "document_store")
            metadata_path = os.path.join(store_dir, "metadata.json")

            import json
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "document_count": len(self.document_store.get_all_documents()),
                    "embedding_model": self.config.get('embedding.default_model'),
                    "processed_content_used": True,  # Flag to indicate processed content was used
                    "last_updated": self._get_timestamp()
                }, f, indent=2)

            return True

        except Exception as e:
            debug(self.config, f"Error adding documents to Haystack document store: {str(e)}")
            print(f"Error adding documents: {str(e)}")
            return False
            
    def build_retrieval_pipeline(self, workspace: str) -> bool:
        """
        Build a retrieval pipeline for a workspace
        
        Args:
            workspace (str): Target workspace
            
        Returns:
            bool: Success flag
        """
        try:
            # Import Haystack pipeline components
            from haystack import Pipeline
            
            # Ensure document store is initialized
            if not hasattr(self, 'document_store') or not hasattr(self, 'retriever'):
                if not self.init_document_store(workspace):
                    return False
            
            # Create retrieval pipeline
            retrieval_pipeline = Pipeline()
            retrieval_pipeline.add_component("retriever", self.retriever)
            
            # Store the pipeline for this workspace
            self.pipelines[workspace] = {
                "retrieval": retrieval_pipeline
            }
            
            return True
            
        except Exception as e:
            debug(self.config, f"Error building Haystack retrieval pipeline: {str(e)}")
            print(f"Error building retrieval pipeline: {str(e)}")
            return False
    
    def query(self, workspace: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query documents from the document store
        
        Args:
            workspace (str): Target workspace
            query_text (str): Query string
            k (int): Number of documents to retrieve
            
        Returns:
            List[Dict]: Retrieved documents with relevance scores
        """
        debug(self.config, f"Querying Haystack document store with k={k}")
        
        # Initialize retrieval pipeline if needed
        if workspace not in self.pipelines or "retrieval" not in self.pipelines[workspace]:
            if not self.build_retrieval_pipeline(workspace):
                return []
        
        try:
            # Get the retrieval pipeline
            pipeline = self.pipelines[workspace]["retrieval"]
            
            # Set top_k parameter on the retriever
            self.retriever.top_k = k
            
            # Run the pipeline
            result = pipeline.run(query=query_text)
            documents = result["documents"]
            
            # Convert to output format
            results = []
            for doc in documents:
                for doc in documents:
                    # Ensure we preserve the original source path
                    meta = doc.meta.copy() if hasattr(doc, 'meta') and doc.meta else {}

                    # Haystack sometimes stores source under different keys
                    if 'source' not in meta and 'file_path' in meta:
                        meta['source'] = meta['file_path']
                    elif 'source' not in meta and 'name' in meta:
                        meta['source'] = meta['name']

                    results.append({
                        "content": doc.content,
                        "metadata": meta,
                        "relevance_score": doc.score if hasattr(doc, 'score') else 1.0
                    })
            
            return results
            
        except Exception as e:
            debug(self.config, f"Error querying Haystack document store: {str(e)}")
            print(f"Error during query: {str(e)}")
            return []
    
    def _get_timestamp(self):
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.now().isoformat()
