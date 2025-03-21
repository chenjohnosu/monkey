
"""
LlamaIndex integration for document processing and vector search
"""

import os
import datetime
import json
from typing import List, Dict, Any, Optional
from core.engine.utils import ensure_dir
from core.engine.logging import debug_print
import shutil
from datetime import datetime

class LlamaIndexConnector:
    """Provides integration with LlamaIndex for document processing and retrieval"""

    def __init__(self, config):
        """Initialize LlamaIndex connector with configuration"""
        self.config = config
        self.index = None
        self.storage_context = None

        # Explicitly configure embedding model to use local embeddings
        self._configure_local_embeddings()

        debug_print(config, "LlamaIndex connector initialized")

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

            print(f"Configuring LlamaIndex with embedding model: {model_name}")

            # Create local embedding model
            try:
                embed_model = HuggingFaceEmbedding(model_name=model_name)
                Settings.embed_model = embed_model
                print(f"Successfully configured embedding model: {model_name}")
            except Exception as e:
                print(f"Error creating main embedding model: {str(e)}")
                # Fall back to a simpler model
                try:
                    print("Falling back to bge-small-en model")
                    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
                    print("Successfully configured fallback embedding model")
                except Exception as e2:
                    print(f"Error creating fallback embedding model: {str(e2)}")

        except Exception as e:
            print(f"Error configuring local embeddings: {str(e)}")
            import traceback
            print(traceback.format_exc())

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
            # Import LlamaIndex components
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.core import Settings
            from llama_index.core.vector_stores.simple import SimpleVectorStore

            # Get vector store directory
            vector_dir = os.path.join("data", workspace, "vector_store")
            ensure_dir(vector_dir)
            print(f"Vector store directory: {vector_dir}")

            # Check for vector store files with different prefixes
            docstore_path = os.path.join(vector_dir, "docstore.json")
            index_store_path = os.path.join(vector_dir, "index_store.json")
            vector_store_path = os.path.join(vector_dir, "vector_store.json")

            # Look for prefixed vector store files
            prefixed_vector_files = [f for f in os.listdir(vector_dir)
                                     if f.endswith('__vector_store.json') or
                                     f.endswith('_vector_store.json')]

            # If we have prefixed files but no vector_store.json, do a quick fix
            if prefixed_vector_files and not os.path.exists(vector_store_path):
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
                    shutil.copy2(
                        os.path.join(vector_dir, largest_file),
                        vector_store_path
                    )

            # Check if files exist
            has_docstore = os.path.exists(docstore_path)
            has_index_store = os.path.exists(index_store_path)
            has_vector_store = os.path.exists(vector_store_path)

            files_exist = has_docstore and has_index_store and has_vector_store

            print(f"Vector store files exist: {files_exist}")
            print(f"  - docstore.json: {has_docstore}")
            print(f"  - index_store.json: {has_index_store}")
            print(f"  - vector_store.json: {has_vector_store}")

            # Create storage context
            if not files_exist:
                print("Creating new vector store")

                # Explicitly use SimpleVectorStore with consistent collection name
                vector_store = SimpleVectorStore(collection_name="vector_store")
                self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # Create an empty index with this storage context
                self.index = VectorStoreIndex(
                    [],
                    storage_context=self.storage_context
                )

                # Persist the empty index
                self.storage_context.persist(persist_dir=vector_dir)
                print("Initialized new empty vector store")

                # Verify files were created
                print("Checking if files were created:")
                print(f"  - docstore.json: {os.path.exists(docstore_path)}")
                print(f"  - index_store.json: {os.path.exists(index_store_path)}")
                print(f"  - vector_store.json: {os.path.exists(vector_store_path)}")
            else:
                # Try to load existing index
                print("Loading existing vector store")
                from llama_index.core import load_index_from_storage

                try:
                    self.storage_context = StorageContext.from_defaults(persist_dir=vector_dir)
                    self.index = load_index_from_storage(self.storage_context)

                    # Check if index was loaded properly
                    if self.index and hasattr(self.index, 'docstore') and self.index.docstore:
                        doc_count = len(self.index.docstore.docs) if self.index.docstore.docs else 0
                        print(f"Successfully loaded vector store with {doc_count} documents")
                    else:
                        print("Warning: Index loaded but may be empty or invalid")
                except Exception as e:
                    print(f"Error loading vector store: {str(e)}")
                    print("Creating new vector store due to load error")

                    # Explicitly use SimpleVectorStore with consistent collection name
                    vector_store = SimpleVectorStore(collection_name="vector_store")
                    self.storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    # Create an empty index with this storage context
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

    def add_documents(self, workspace: str, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store

        Args:
            workspace (str): Target workspace
            documents (List[Dict]): List of document objects with content and metadata

        Returns:
            bool: Success flag
        """
        print(f"Adding {len(documents)} documents to LlamaIndex vector store")

        try:
            # Import necessary LlamaIndex components
            from llama_index.core.schema import Document as LlamaDocument
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.core import Settings
            from llama_index.core.vector_stores.simple import SimpleVectorStore

            # Get vector store directory
            vector_dir = os.path.join("data", workspace, "vector_store")
            ensure_dir(vector_dir)

            # Ensure vector store is initialized
            if self.index is None:
                print("Index not initialized, initializing now")
                if not self.init_vector_store(workspace):
                    print("Failed to initialize vector store")
                    return False

            # Convert documents to LlamaIndex format with processed content
            llama_docs = []
            print(f"Converting {len(documents)} documents to LlamaIndex format")

            # Track source paths for deduplication
            document_sources = {}

            for i, doc in enumerate(documents):
                # Extract content and metadata - EXPLICITLY USE PROCESSED CONTENT WITH FALLBACK
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
                    text=content,  # Using processed content with stop words removed
                    metadata=metadata
                )
                llama_docs.append(llama_doc)

                # Print sample metadata for debugging
                if i == 0:
                    print(f"Sample document metadata: {metadata}")
                    print(f"Using processed content: {content[:100]}..." if len(content) > 100 else content)

            print(f"Created {len(llama_docs)} LlamaIndex documents")

            # Print document count before adding
            doc_count_before = 0
            if self.index and hasattr(self.index, 'docstore') and self.index.docstore:
                doc_count_before = len(self.index.docstore.docs) if self.index.docstore.docs else 0

            print(f"Documents in index before adding: {doc_count_before}")

            # Check if we have a valid index with documents
            if (self.index and hasattr(self.index, 'docstore') and
                    self.index.docstore and self.index.docstore.docs and doc_count_before > 0):
                print(f"Adding documents to existing index with {doc_count_before} documents")

                # Check for duplicate documents by source path
                existing_sources = set()
                for doc_id, doc in self.index.docstore.docs.items():
                    if hasattr(doc, 'metadata') and doc.metadata and 'source' in doc.metadata:
                        existing_sources.add(doc.metadata['source'])

                # Filter out documents that already exist in the index
                new_docs_to_add = []
                for doc in llama_docs:
                    if doc.metadata.get('source') not in existing_sources:
                        new_docs_to_add.append(doc)
                    else:
                        print(f"Skipping duplicate document: {doc.metadata.get('source')}")

                print(f"Found {len(new_docs_to_add)} new documents out of {len(llama_docs)} total")

                # Add only new documents to existing index
                if new_docs_to_add:
                    for doc in new_docs_to_add:
                        self.index.insert(doc)
                else:
                    print("No new documents to add")

                # Get storage context from existing index
                if hasattr(self.index, 'storage_context'):
                    storage_context = self.index.storage_context
                else:
                    # Create a new storage context for persistence
                    print("Creating storage context from existing vector store")
                    storage_context = self.storage_context
            else:
                print("Creating new index for documents")
                # Create a new index with consistent vector store naming
                vector_store = SimpleVectorStore(collection_name="vector_store")
                storage_context = StorageContext.from_defaults(vector_store=vector_store)

                # Create new index with all documents
                self.index = VectorStoreIndex.from_documents(
                    llama_docs,
                    storage_context=storage_context
                )

                # Update storage context reference
                self.storage_context = storage_context

            # Check document count after adding
            doc_count_after = len(self.index.docstore.docs) if self.index.docstore.docs else 0
            print(f"Documents in index after adding: {doc_count_after}")

            # Verify documents were added correctly
            if doc_count_after == 0:
                print("ERROR: No documents in index after adding!")
                return False
            elif doc_count_after < doc_count_before + len(
                    new_docs_to_add if 'new_docs_to_add' in locals() else llama_docs):
                expected_count = doc_count_before + len(
                    new_docs_to_add if 'new_docs_to_add' in locals() else llama_docs)
                print(f"WARNING: Document count mismatch. Expected {expected_count}, got {doc_count_after}")

            # Persist the updated index
            print("Persisting index to disk")
            if storage_context:
                storage_context.persist(persist_dir=vector_dir)
            else:
                print("ERROR: No storage context available for persistence!")
                return False

            # Verify files were created
            docstore_path = os.path.join(vector_dir, "docstore.json")
            index_store_path = os.path.join(vector_dir, "index_store.json")
            vector_store_path = os.path.join(vector_dir, "vector_store.json")

            print("Checking vector store files after adding documents:")
            print(
                f"  - docstore.json: {os.path.exists(docstore_path)} - Size: {os.path.getsize(docstore_path) if os.path.exists(docstore_path) else 0} bytes")
            print(
                f"  - index_store.json: {os.path.exists(index_store_path)} - Size: {os.path.getsize(index_store_path) if os.path.exists(index_store_path) else 0} bytes")
            print(
                f"  - vector_store.json: {os.path.exists(vector_store_path)} - Size: {os.path.getsize(vector_store_path) if os.path.exists(vector_store_path) else 0} bytes")

            # Check for any prefixed vector store files and migrate if needed
            prefixed_files = [f for f in os.listdir(vector_dir) if f.endswith('__vector_store.json')]
            if prefixed_files and not os.path.exists(vector_store_path):
                print(f"Found prefixed vector store files but no vector_store.json. Copying largest file...")
                largest_file = None
                largest_size = 0
                for file in prefixed_files:
                    file_path = os.path.join(vector_dir, file)
                    file_size = os.path.getsize(file_path)
                    if file_size > largest_size:
                        largest_size = file_size
                        largest_file = file

                if largest_file:
                    print(f"Copying {largest_file} to vector_store.json")
                    import shutil
                    shutil.copy2(
                        os.path.join(vector_dir, largest_file),
                        vector_store_path
                    )

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
                    "new_documents": len(new_docs_to_add if 'new_docs_to_add' in locals() else llama_docs),
                    "deduplication_applied": 'new_docs_to_add' in locals()
                }, f, indent=2)

            print(f"Successfully added documents to index")
            return True

        except Exception as e:
            print(f"Error adding documents to LlamaIndex vector store: {str(e)}")
            import traceback
            print(traceback.format_exc())
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
        print(f"Querying LlamaIndex vector store with k={k}, query: '{query_text}'")

        # Initialize or load the vector store if needed
        if self.index is None:
            print("Index not initialized, loading vector store")
            if not self.init_vector_store(workspace):
                print("Failed to load vector store for query")
                return []

        try:
            # Check if the index has documents
            if not hasattr(self.index, 'docstore') or not self.index.docstore.docs:
                print("WARNING: Index has no documents!")
                return []

            doc_count = len(self.index.docstore.docs)
            print(f"Index has {doc_count} documents")

            # Perform the query
            print(f"Creating retriever with similarity_top_k={k}")
            retriever = self.index.as_retriever(similarity_top_k=k)

            print("Retrieving nodes for query")
            nodes = retriever.retrieve(query_text)
            print(f"Retrieved {len(nodes)} nodes")

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
                    print(f"Sample node - Score: {node.score if hasattr(node, 'score') else 'N/A'}")
                    print(f"Sample node - Metadata: {node.metadata}")
                    print(f"Sample node - Text preview: {node.text[:100]}...")

            return results

        except Exception as e:
            print(f"Error querying LlamaIndex vector store: {str(e)}")
            import traceback
            print(traceback.format_exc())
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