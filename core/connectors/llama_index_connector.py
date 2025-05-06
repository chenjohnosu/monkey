
"""
LlamaIndex integration for document processing and vector search
"""

import os
import datetime
import json
from typing import List, Dict, Any, Optional
from core.engine.utils import ensure_dir
from core.engine.logging import debug,error,info,warning,trace,debug
import shutil
from datetime import datetime


def _ensure_nltk_initialized():
    """
    Ensure NLTK is properly initialized before LlamaIndex uses it
    This prevents the 'module nltk has no attribute data' error
    """
    try:
        import nltk
        import sys

        # Check if nltk.data is already in sys.modules
        if 'nltk.data' not in sys.modules:
            try:
                # Try importing the data module explicitly
                import nltk.data
            except ImportError:
                # If that fails, download a minimal dataset which will initialize the module
                print("Initializing NLTK data module...")
                nltk.download('punkt', quiet=True)
                # Try importing again after download
                import nltk.data

        # Verify that nltk.data exists
        if not hasattr(nltk, 'data'):
            print("Warning: NLTK data module not properly initialized")
            return False

        return True
    except Exception as e:
        print(f"Error initializing NLTK: {str(e)}")
        return False

class LlamaIndexConnector:
    """Provides integration with LlamaIndex for document processing and retrieval"""

    def __init__(self, config):
        """Initialize LlamaIndex connector with configuration"""
        self.config = config
        self.index = None
        self.storage_context = None

        # Explicitly configure embedding model to use local embeddings
        self._configure_local_embeddings()

        debug(config, "LlamaIndex connector initialized")

    def _configure_local_embeddings(self):
        """
        Configure local embedding model
        """
        try:
            from llama_index.core import Settings
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Get embedding model from config
            embedding_model_name = self.config.get('embedding.default_model', 'multilingual-e5')

            # Select appropriate local embedding model
            if embedding_model_name == "multilingual-e5":
                model_name = "intfloat/multilingual-e5-large"
            elif embedding_model_name == "mixbread":
                model_name = "mixedbread-ai/mxbai-embed-large-v1"
            elif embedding_model_name == "bge":
                model_name = "BAAI/bge-m3"
            else:
                # Fallback to a reliable multilingual model
                model_name = "intfloat/multilingual-e5-large"

            info(f"Configuring LlamaIndex with embedding model: {model_name}")

            # Create local embedding model
            try:
                embed_model = HuggingFaceEmbedding(model_name=model_name)
                Settings.embed_model = embed_model
                info(f"Successfully configured embedding model: {model_name}")
            except Exception as e:
                error(f"Error creating main embedding model: {str(e)}")
                # Fall back to a simpler model
                try:
                    warning("Falling back to bge-small-en model")
                    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
                    info("Successfully configured fallback embedding model")
                except Exception as e2:
                    error(f"Error creating fallback embedding model: {str(e2)}")

        except Exception as e:
            error(f"Error configuring local embeddings: {str(e)}")
            import traceback
            trace(traceback.format_exc())

    def add_documents(self, workspace: str, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store

        Args:
            workspace (str): Target workspace
            documents (List[Dict]): List of document objects with content and metadata

        Returns:
            bool: Success flag
        """
        info(f"Adding {len(documents)} documents to LlamaIndex vector store")

        try:
            # Import necessary LlamaIndex components
            from llama_index.core.schema import Document as LlamaDocument
            from llama_index.core import VectorStoreIndex, load_index_from_storage

            # Get vector store directory
            vector_dir = os.path.join("data", workspace, "vector_store")
            ensure_dir(vector_dir)

            # Ensure vector store is initialized
            if self.index is None:
                warning("Index not initialized, initializing now")
                if not self.init_vector_store(workspace):
                    error("Failed to initialize vector store")
                    return False

            # Convert documents to LlamaIndex format with processed content
            llama_docs = []
            info(f"Converting {len(documents)} documents to LlamaIndex format")

            # Track source paths for deduplication
            document_sources = {}

            for i, doc in enumerate(documents):
                # Extract content and metadata - USE PROCESSED CONTENT WITH FALLBACK
                content = doc.get("processed_content", doc.get("content", ""))
                metadata = doc.get("metadata", {}).copy()
                source = metadata.get("source", f"doc_{i}")

                # Track the source path
                document_sources[source] = i

                # Sanitize metadata (exclude non-serializable values)
                for key in list(metadata.keys()):
                    if not isinstance(metadata[key], (str, int, float, bool, list, dict)):
                        del metadata[key]

                # Create LlamaDocument with processed content
                llama_doc = LlamaDocument(
                    text=content,
                    metadata=metadata
                )
                llama_docs.append(llama_doc)

            info(f"Created {len(llama_docs)} LlamaIndex documents")

            # Print document count before adding
            doc_count_before = 0
            if self.index and hasattr(self.index, 'docstore') and self.index.docstore:
                doc_count_before = len(self.index.docstore.docs) if self.index.docstore.docs else 0

            info(f"Documents in index before adding: {doc_count_before}")

            # Check if we have a valid index with documents
            if doc_count_before > 0:
                info(f"Adding documents to existing index with {doc_count_before} documents")

                # Check for duplicate documents by source path
                existing_sources = set()
                for doc_id, doc in self.index.docstore.docs.items():
                    if hasattr(doc, 'metadata') and doc.metadata and 'source' in doc.metadata:
                        existing_sources.add(doc.metadata['source'])

                # Filter out documents that already exist
                new_docs_to_add = []
                for doc in llama_docs:
                    if doc.metadata.get('source') not in existing_sources:
                        new_docs_to_add.append(doc)
                    else:
                        info(f"Skipping duplicate document: {doc.metadata.get('source')}")

                print(f"Found {len(new_docs_to_add)} new documents out of {len(llama_docs)} total")

                # Add only new documents to existing index
                if new_docs_to_add:
                    for doc in new_docs_to_add:
                        self.index.insert(doc)
                else:
                    print("No new documents to add")

            else:
                info("Creating new index for documents")
                # Use centralized storage context creation
                self.storage_context = self._get_storage_context(workspace, create_new=True)

                # Create new index with all documents
                self.index = VectorStoreIndex.from_documents(
                    llama_docs,
                    storage_context=self.storage_context
                )

            # Verify documents were added correctly
            doc_count_after = len(self.index.docstore.docs) if hasattr(self.index,
                                                                       'docstore') and self.index.docstore.docs else 0
            print(f"Documents in index after adding: {doc_count_after}")

            # Persist the updated index
            info("Persisting index to disk")
            if self.storage_context:
                self.storage_context.persist(persist_dir=vector_dir)
                info(f"Index persisted successfully to {vector_dir}")
            else:
                error("ERROR: No storage context available for persistence!")
                return False

            # Save additional metadata for debugging
            metadata_path = os.path.join(vector_dir, "custom_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                import datetime
                import json
                json.dump({
                    "created": datetime.datetime.now().isoformat(),
                    "document_count": len(documents),
                    "embedding_model": self.config.get('embedding.default_model'),
                    "documents_added": doc_count_after,
                    "documents_before": doc_count_before,
                    "delta": doc_count_after - doc_count_before,
                    "new_documents": len(new_docs_to_add if 'new_docs_to_add' in locals() else llama_docs)
                }, f, indent=2)

            print(f"Successfully added documents to index")
            return True

        except Exception as e:
            error(f"Error adding documents to LlamaIndex vector store: {str(e)}")
            import traceback
            trace(traceback.format_exc())
            return False

    def query(self, workspace: str, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Query documents from the vector store

        Args:
            workspace (str): Target workspace
            query_text (str): Query string
            k (int): Number of documents to retrieve

        Returns:
            List[Dict]: Retrieved documents with relevance scores
        """
        info(f"Querying LlamaIndex vector store with k={k}, query: '{query_text}'")

        # Initialize or load the vector store if needed
        if self.index is None:
            warning("Index not initialized, loading vector store")
            if not self.init_vector_store(workspace):
                error("Failed to load vector store for query")
                return []

        try:
            # Check if the index has documents
            if not hasattr(self.index, 'docstore') or not self.index.docstore.docs:
                warning("WARNING: Index has no documents!")
                return []

            doc_count = len(self.index.docstore.docs)
            info(f"Index has {doc_count} documents")

            # Perform the query
            info(f"Creating retriever with similarity_top_k={k}")
            retriever = self.index.as_retriever(similarity_top_k=k)

            info("Retrieving nodes for query")
            nodes = retriever.retrieve(query_text)
            info(f"Retrieved {len(nodes)} nodes")

            # Convert to output format
            results = []
            for i, node in enumerate(nodes):
                result = {
                    "content": node.text,
                    "metadata": node.metadata,
                    "relevance_score": node.score if hasattr(node, 'score') else 1.0
                }
                results.append(result)

                # Print sample node info
                if i == 0:
                    debug(f"Sample node - Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                    debug(f"Sample node - Metadata: {node.metadata}")
                    debug(f"Sample node - Text preview: {node.text[:100]}...")

            return results

        except Exception as e:
            error(f"Error querying LlamaIndex vector store: {str(e)}")
            import traceback
            trace(traceback.format_exc())
            return []

    def inspect_index_store(self, workspace):
        """
        Thoroughly inspect the LlamaIndex index store for a given workspace

        Args:
            workspace (str): Target workspace to inspect

        Returns:
            Dict: Detailed information about the index store
        """
        print(f"\nInspecting LlamaIndex Index Store for Workspace: {workspace}")

        # Define paths for key LlamaIndex files
        vector_dir = os.path.join("data", workspace, "vector_store")
        index_store_path = os.path.join(vector_dir, "index_store.json")
        docstore_path = os.path.join(vector_dir, "docstore.json")
        vector_store_path = os.path.join(vector_dir, "vector_store.json")

        # Debug print paths
        print("\nVector Store Paths:")
        print(f"  Directory: {vector_dir}")
        print(f"  Index Store: {index_store_path}")
        print(f"  Docstore: {docstore_path}")
        print(f"  Vector Store: {vector_store_path}")

        # Verify directory contents
        print("\nDirectory Contents:")
        try:
            files = os.listdir(vector_dir)
            for file in files:
                print(f"  {file}")
        except Exception as e:
            print(f"Error listing directory: {str(e)}")

        # Check file contents comprehensively
        files_to_check = {
            "index_store.json": index_store_path,
            "docstore.json": docstore_path,
            "vector_store.json": vector_store_path
        }

        # Comprehensive file analysis
        for file_name, file_path in files_to_check.items():
            if os.path.exists(file_path):
                print(f"\nAnalyzing {file_name}:")
                print(f"  File Size: {os.path.getsize(file_path)} bytes")

                try:
                    with open(file_path, 'r') as f:
                        try:
                            file_data = json.load(f)

                            # Comprehensive key investigation
                            print("  Full JSON Structure:")
                            for key, value in file_data.items():
                                if isinstance(value, dict):
                                    print(f"    {key}: {len(value)} entries")
                                    # Print first few keys if not too many
                                    if len(value) > 0:
                                        print("      Sample keys:", list(value.keys())[:5])
                                elif isinstance(value, list):
                                    print(f"    {key}: {len(value)} items")
                                else:
                                    print(f"    {key}: {type(value).__name__}")

                        except json.JSONDecodeError as json_err:
                            print(f"  JSON Parsing Error: {json_err}")

                except Exception as e:
                    print(f"  Error reading file: {str(e)}")
            else:
                print(f"{file_name} not found")

        # Try to load index and get details
        print("\nAttempting to Load Index:")
        try:
            from llama_index.core import load_index_from_storage, StorageContext

            # Create storage context
            storage_context = StorageContext.from_defaults(persist_dir=vector_dir)

            # Load index
            index = load_index_from_storage(storage_context)

            # Check index details
            print("  Index Loaded Successfully!")

            # Check document store
            if hasattr(index, 'docstore'):
                print("  Docstore Details:")
                try:
                    docs = index.docstore.docs
                    print(f"    Total Documents: {len(docs)}")

                    # Print sample document details
                    if docs:
                        sample_doc_id = list(docs.keys())[0]
                        sample_doc = docs[sample_doc_id]
                        print("    Sample Document:")
                        print(f"      Keys: {list(sample_doc.__dict__.keys())}")
                        if hasattr(sample_doc, 'text'):
                            print(f"      Text Preview: {sample_doc.text[:200]}...")
                except Exception as e:
                    print(f"    Error accessing docstore: {str(e)}")

        except Exception as e:
            print(f"  Error loading index: {str(e)}")
            import traceback
            traceback.print_exc()

        return None  # Placeholder return

    def _get_storage_context(self, workspace: str, create_new: bool = False):
        """
        Centralized factory method for creating or loading storage contexts

        Args:
            workspace (str): Target workspace
            create_new (bool): Whether to create a new storage context even if files exist

        Returns:
            StorageContext: The created or loaded storage context
        """
        from llama_index.core import StorageContext
        from llama_index.core.vector_stores.simple import SimpleVectorStore

        # Get vector store directory
        vector_dir = os.path.join("data", workspace, "vector_store")
        ensure_dir(vector_dir)

        # Check if vector store files exist
        docstore_path = os.path.join(vector_dir, "docstore.json")
        index_store_path = os.path.join(vector_dir, "index_store.json")
        vector_store_path = os.path.join(vector_dir, "vector_store.json")

        files_exist = (os.path.exists(docstore_path) and
                       os.path.exists(index_store_path) and
                       os.path.exists(vector_store_path))

        if files_exist and not create_new:
            # Load existing storage context
            print(f"Loading existing storage context from {vector_dir}")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=vector_dir)
                return storage_context
            except Exception as e:
                print(f"Error loading storage context: {str(e)}")
                # Fall through to create new context

        # Create new storage context with consistent collection name
        print(f"Creating new storage context with consistent collection name")
        vector_store = SimpleVectorStore(collection_name="vector_store")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        return storage_context

    def init_vector_store(self, workspace: str) -> bool:
        """
        Initialize a vector store for a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            bool: Success flag
        """
        print(f"Initializing LlamaIndex vector store for workspace: {workspace}")

        try:
            _ensure_nltk_initialized()

            # Import LlamaIndex components
            from llama_index.core import VectorStoreIndex, load_index_from_storage

            # Get vector store directory
            vector_dir = os.path.join("data", workspace, "vector_store")
            ensure_dir(vector_dir)
            print(f"Vector store directory: {vector_dir}")

            # Check for vector store files
            docstore_path = os.path.join(vector_dir, "docstore.json")
            index_store_path = os.path.join(vector_dir, "index_store.json")
            vector_store_path = os.path.join(vector_dir, "vector_store.json")

            # Look for prefixed vector store files and handle them
            self._handle_prefixed_vector_files(vector_dir, vector_store_path)

            # Check if files exist after possible migration
            files_exist = (os.path.exists(docstore_path) and
                           os.path.exists(index_store_path) and
                           os.path.exists(vector_store_path))

            print(f"Vector store files exist: {files_exist}")

            if not files_exist:
                # Create new empty index with storage context
                self.storage_context = self._get_storage_context(workspace, create_new=True)
                self.index = VectorStoreIndex(
                    [],
                    storage_context=self.storage_context
                )

                # Persist the empty index
                self.storage_context.persist(persist_dir=vector_dir)
                print("Initialized new empty vector store")
            else:
                # Load existing index with centralized storage context
                try:
                    self.storage_context = self._get_storage_context(workspace)
                    self.index = load_index_from_storage(self.storage_context)

                    # Check if index was loaded properly
                    if self.index and hasattr(self.index, 'docstore') and self.index.docstore:
                        doc_count = len(self.index.docstore.docs) if self.index.docstore.docs else 0
                        print(f"Successfully loaded vector store with {doc_count} documents")
                    else:
                        print("Warning: Index loaded but may be empty or invalid")
                except Exception as e:
                    print(f"Error loading vector store: {str(e)}")

                    # Create new vector store due to load error
                    self.storage_context = self._get_storage_context(workspace, create_new=True)
                    self.index = VectorStoreIndex(
                        [],
                        storage_context=self.storage_context
                    )

                    # Persist the empty index
                    self.storage_context.persist(persist_dir=vector_dir)
                    print("Initialized new empty vector store after load failure")

            return True

        except Exception as e:
            print(f"Error initializing LlamaIndex vector store: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False

    def _handle_prefixed_vector_files(self, vector_dir, vector_store_path):
        """
        Handle prefixed vector store files by copying the largest to vector_store.json

        Args:
            vector_dir (str): Vector store directory
            vector_store_path (str): Path to the expected vector_store.json file
        """
        # Skip if vector_store.json already exists
        if os.path.exists(vector_store_path):
            return

        prefixed_vector_files = [f for f in os.listdir(vector_dir)
                                 if f.endswith('__vector_store.json') or
                                 f.endswith('_vector_store.json')]

        if not prefixed_vector_files:
            return

        print(f"Found prefixed vector store files but no vector_store.json. Attempting quick fix...")

        # Use the largest prefixed file
        largest_file = None
        largest_size = 0

        for file in prefixed_vector_files:
            file_path = os.path.join(vector_dir, file)
            file_size = os.path.getsize(file_path)
            if file_size > largest_size:
                largest_size = file_size
                largest_file = file

        if largest_file:
            print(f"Using {largest_file} as vector_store.json")
            import shutil
            shutil.copy2(
                os.path.join(vector_dir, largest_file),
                vector_store_path
            )