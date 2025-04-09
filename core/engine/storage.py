"""
Vector store management using LlamaIndex and Haystack
"""

import os
import json
import hashlib
import datetime
import shutil
from core.engine.utils import ensure_dir
from core.engine.logging import debug_print,info,error,warning,debug
from core.connectors.connector_factory import ConnectorFactory
from typing import List, Dict, Any

class StorageManager:
    """Manages document storage and vector databases using LlamaIndex or Haystack"""

    def __init__(self, config):
        """Initialize the storage manager"""
        self.config = config
        self.vector_stores = {}  # Cache for loaded vector stores
        self.factory = ConnectorFactory(config)
        self.connector = self.factory.get_vector_store_connector()
        debug_print(config,
                    "Storage manager initialized with connector type: " + self.config.get('storage.vector_store'))

    def add_document(self, workspace, source_path, content, processed_content, metadata):
        """
        Add a document to storage and vector store

        Args:
            workspace (str): Target workspace
            source_path (str): Document source path
            content (str): Original document content
            processed_content (str): Processed document content
            metadata (dict): Document metadata
        """
        debug_print(self.config, f"Adding document to workspace: {workspace}, source: {source_path}")

        # First, save document to document storage
        doc_dir = os.path.join("data", workspace, "documents")
        ensure_dir(doc_dir)

        # Create safe filename from source path
        filename = hashlib.md5(source_path.encode('utf-8')).hexdigest() + '.json'
        filepath = os.path.join(doc_dir, filename)

        # Create document object
        document = {
            'content': content,
            'processed_content': processed_content,
            'metadata': metadata
        }

        # Save document to file
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(document, file, indent=2)

        # Add to vector store
        debug_print(self.config, f"Adding document to vector store: {source_path}")
        self._add_to_vector_store(workspace, document)

    def remove_document(self, workspace, source_path):
        """
        Remove a document from storage

        Args:
            workspace (str): Target workspace
            source_path (str): Document source path

        Returns:
            bool: Success flag
        """
        debug_print(self.config, f"Removing document from workspace: {workspace}, source: {source_path}")

        # Create safe filename from source path
        import hashlib
        filename = hashlib.md5(source_path.encode('utf-8')).hexdigest() + '.json'
        doc_dir = os.path.join("data", workspace, "documents")
        filepath = os.path.join(doc_dir, filename)

        # Remove file if it exists
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                debug_print(self.config, f"Removed document file: {filepath}")
                return True
            except Exception as e:
                debug_print(self.config, f"Error removing document file: {str(e)}")
                return False
        else:
            debug_print(self.config, f"Document file not found: {filepath}")
            return False

    def add_documents(self, workspace: str, documents: List[Dict[str, Any]]) -> bool:
        """Add documents to the vector store"""
        debug_print(self.config, f"Adding {len(documents)} documents to LlamaIndex vector store")

        try:
            # Import necessary LlamaIndex components
            from llama_index.core.schema import Document as LlamaDocument
            from llama_index.core import VectorStoreIndex, StorageContext
            from llama_index.core.vector_stores.simple import SimpleVectorStore

            # IMPORTANT: Use consistent vector store configuration
            vector_store = SimpleVectorStore(collection_name="vector_store")

            # Get vector store directory
            vector_dir = os.path.join("data", workspace, "vector_store")
            ensure_dir(vector_dir)

            # Ensure vector store is initialized
            if self.index is None:
                if not self.init_vector_store(workspace):
                    debug_print(self.config, "Failed to initialize vector store")
                    return False

            # Convert documents to LlamaIndex format
            llama_docs = []
            # ... (rest of the conversion code)

            # Add documents to the index with explicit vector store
            if not self.index.docstore.docs:
                # If index is empty, create a new one with explicit vector store
                storage_context = StorageContext.from_defaults(vector_store=vector_store)
                self.index = VectorStoreIndex.from_documents(
                    llama_docs,
                    storage_context=storage_context
                )
            else:
                # Add documents to existing index
                for doc in llama_docs:
                    self.index.insert(doc)

            # Persist the updated index
            if self.storage_context:
                self.storage_context.persist(persist_dir=vector_dir)

            return True
        except Exception as e:
            debug_print(self.config, f"Error adding documents: {str(e)}")
            return False

    def _add_to_vector_store(self, workspace, document):
        """
        Add a document to the vector store

        Args:
            workspace (str): Target workspace
            document (dict): Document object
        """
        vector_store_type = self.config.get('storage.vector_store')
        debug_print(self.config, f"Adding document to {vector_store_type} vector store")

        # Create list with single document
        documents = [document]

        # Use the appropriate connector
        try:
            success = self.connector.add_documents(workspace, documents)
            if not success:
                debug_print(self.config, "Failed to add document to vector store")
            return success
        except Exception as e:
            debug_print(self.config, f"Error adding document to vector store: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return False

    def update_vector_store(self, workspace, documents=None):
        """
        Update the vector store with new or modified documents without rebuilding from scratch

        Args:
            workspace (str): Target workspace
            documents (List[Dict], optional): Specific documents to update. If None, updates all documents.

        Returns:
            bool: Success flag
        """
        debug_print(self.config, f"Updating vector store for workspace: {workspace}")

        try:
            # Get documents to update (either provided or load all)
            docs_to_update = documents or self.get_documents(workspace)

            if not docs_to_update:
                print(f"No documents to update for workspace '{workspace}'")
                return False

            print(f"Updating vector store with {len(docs_to_update)} documents...")

            # Initialize vector store if needed
            vector_store_initialized = self.load_vector_store(workspace)

            if not vector_store_initialized:
                print("Vector store not initialized. Creating new vector store...")
                return self.create_vector_store(workspace)

            # Get vector store type
            vector_store_type = self.config.get('storage.vector_store')

            # Use appropriate connector update method
            if vector_store_type == 'llama_index':
                return self._update_llama_index_documents(workspace, docs_to_update)
            elif vector_store_type == 'haystack':
                return self._update_haystack_documents(workspace, docs_to_update)
            else:
                # Fallback to create_vector_store for unsupported backends
                print(f"Incremental updates not supported for {vector_store_type}. Rebuilding vector store...")
                return self.create_vector_store(workspace)

        except Exception as e:
            debug_print(self.config, f"Error updating vector store: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return False

    def get_documents(self, workspace):
        """
        Get all documents in a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            list: Document objects
        """
        debug_print(self.config, f"Getting documents from workspace: {workspace}")

        # Verify data and documents directory exists
        import os
        workspace_dir = os.path.join("data", workspace)
        documents_dir = os.path.join(workspace_dir, "documents")

        # Detailed diagnostics
        debug(f"Workspace Document Retrieval:")
        debug(f"  Workspace Directory: {workspace_dir}")
        debug(f"  Documents Directory: {documents_dir}")

        # Check directory existence
        if not os.path.exists(workspace_dir):
            error(f"ERROR: Workspace directory does not exist: {workspace_dir}")
            return []

        if not os.path.exists(documents_dir):
            error(f"ERROR: Documents directory does not exist: {documents_dir}")
            # Attempt to create the directory
            try:
                os.makedirs(documents_dir)
                info(f"Created documents directory: {documents_dir}")
            except Exception as e:
                error(f"Failed to create documents directory: {str(e)}")
                return []

        documents = []

        # List files in the documents directory
        try:
            document_files = [f for f in os.listdir(documents_dir) if f.endswith('.json')]
            print(f"  JSON Files in Documents Directory: {len(document_files)}")

            # If no files, provide more context
            if not document_files:
                warning("  Warning: No JSON document files found")
                # List all files in the directory to understand why
                all_files = os.listdir(documents_dir)
                if all_files:
                    print("  All files in documents directory:")
                    for file in all_files:
                        print(f"    - {file}")
                else:
                    warning("  No files found in documents directory")
        except Exception as e:
            print(f"  Error listing document files: {str(e)}")
            return []

        # Load documents from files
        for filename in document_files:
            try:
                with open(os.path.join(documents_dir, filename), 'r', encoding='utf-8') as file:
                    document = json.load(file)
                    documents.append(document)
            except Exception as e:
                error(f"Error loading document {filename}: {str(e)}")

        print(f"  Total Documents Loaded: {len(documents)}")

        # If no documents, provide context
        if not documents:
            warning("No documents could be loaded from the workspace")

        return documents

    def query_documents(self, workspace, query, k=5):
        """
        Query documents in a workspace

        Args:
            workspace (str): Target workspace
            query (str): Query string
            k (int): Number of documents to retrieve

        Returns:
            list: Relevant documents
        """
        debug_print(self.config, f"Querying workspace '{workspace}' with k={k}")

        # Use the appropriate connector
        try:
            results = self.connector.query(workspace, query, k)

            # Normalize metadata in results
            for doc in results:
                if 'metadata' in doc:
                    doc['metadata'] = normalize_source_path(doc['metadata'])

            return results
        except Exception as e:
            debug_print(self.config, f"Error querying vector store: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return []

    def get_processed_files(self, workspace):
        """
        Get a dictionary of processed files and their metadata

        Args:
            workspace (str): Target workspace

        Returns:
            dict: Dictionary mapping file paths to metadata
        """
        debug_print(self.config, f"Getting processed files for workspace: {workspace}")

        processed_files = {}
        documents = self.get_documents(workspace)

        for doc in documents:
            source = doc['metadata'].get('source')

            if source:
                # Extract relevant metadata
                metadata = {
                    'last_modified': doc['metadata'].get('last_modified', 0),
                    'content_hash': doc['metadata'].get('content_hash', ''),
                    'language': doc['metadata'].get('language', 'unknown'),
                    'processed_date': doc['metadata'].get('processed_date', '')
                }

                processed_files[source] = metadata

        return processed_files

    def create_vector_store(self, workspace):
        """
        Create or update the vector store for a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            bool: Success flag
        """
        debug_print(self.config, f"Creating vector store for workspace: {workspace}")

        # Get all documents
        documents = self.get_documents(workspace)
        if not documents:
            print(f"No documents found in workspace '{workspace}'")
            return False

        print(f"Building vector store for {len(documents)} source files")

        # Make sure vector store directory exists and is empty
        vector_dir = os.path.join("data", workspace, "vector_store")
        if os.path.exists(vector_dir):
            # Create backup before removing
            backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            if os.path.isdir(vector_dir) and os.listdir(vector_dir):  # Only backup if not empty
                debug_print(self.config, f"Backing up existing vector store to {backup_dir}")
                shutil.copytree(vector_dir, backup_dir)

            # Remove existing directory and recreate
            shutil.rmtree(vector_dir)

        ensure_dir(vector_dir)

        # Use the appropriate connector to batch add all documents
        try:
            # Clear any cached vector store
            self.vector_stores.pop(workspace, None)

            # Add documents to new vector store
            success = self.connector.add_documents(workspace, documents)

            if success:
                # Count actual embeddings created
                vector_store_type = self.config.get('storage.vector_store')
                if vector_store_type == 'llama_index':
                    # Check actual number of documents in the vector store
                    docstore_path = os.path.join(vector_dir, "docstore.json")
                    if os.path.exists(docstore_path):
                        try:
                            with open(docstore_path, 'r') as f:
                                data = json.load(f)
                                if 'docstore/docs' in data:
                                    index_doc_count = len(data['docstore/docs'])
                                    if index_doc_count != len(documents):
                                        print(
                                            f"Note: {len(documents)} source files resulted in {index_doc_count} vector index entries")
                        except:
                            pass

                print(f"Vector store created successfully with {len(documents)} source files")

                # Save vector store metadata
                metadata_path = os.path.join(vector_dir, "metadata.json")
                with open(metadata_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "created": datetime.datetime.now().isoformat(),
                        "source_file_count": len(documents),
                        "embedding_model": self.config.get('embedding.default_model')
                    }, f, indent=2)

            return success

        except Exception as e:
            debug_print(self.config, f"Error creating vector store: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return False

    def load_vector_store(self, workspace):
        """
        Load a vector store for a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            bool: Success flag
        """
        debug_print(self.config, f"Loading vector store for workspace: {workspace}")

        # Check if already loaded
        if workspace in self.vector_stores:
            debug_print(self.config, f"Vector store for workspace '{workspace}' already loaded")
            return True

        # For LlamaIndex and Haystack, we need to initialize the store
        vector_store_type = self.config.get('storage.vector_store')

        try:
            if vector_store_type == 'llama_index':
                success = self.connector.init_vector_store(workspace)
            elif vector_store_type == 'haystack':
                success = self.connector.init_document_store(workspace)
            else:
                # Default behavior - look for existing vector store
                vector_dir = os.path.join("data", workspace, "vector_store")
                metadata_path = os.path.join(vector_dir, "metadata.json")
                success = os.path.exists(metadata_path)

            if success:
                # Store in cache
                self.vector_stores[workspace] = {
                    'loaded': datetime.datetime.now().isoformat(),
                }
                print(f"Vector store loaded for workspace '{workspace}'")

            return success
        except Exception as e:
            debug_print(self.config, f"Error loading vector store: {str(e)}")
            return False

    def get_workspace_stats(self, workspace):
        """
        Get statistics for a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            dict: Workspace statistics
        """
        debug_print(self.config, f"Getting statistics for workspace: {workspace}")

        # Check for document directory
        workspace_dir = os.path.join("data", workspace, "documents")
        if not os.path.exists(workspace_dir):
            return None

        try:
            # Count documents
            documents = self.get_documents(workspace)
            doc_count = len(documents)

            # Count documents by language
            language_counts = {}
            for doc in documents:
                lang = doc['metadata'].get('language', 'unknown')
                language_counts[lang] = language_counts.get(lang, 0) + 1

            # Look for vector store metadata
            vector_dir = os.path.join("data", workspace, "vector_store")
            metadata_path = os.path.join(vector_dir, "metadata.json")

            if os.path.exists(metadata_path):
                # Load metadata if exists
                with open(metadata_path, 'r', encoding='utf-8') as file:
                    metadata = json.load(file)
            else:
                # Create basic metadata
                metadata = {
                    'created': datetime.datetime.now().isoformat(),
                    'document_count': doc_count,
                    'embedding_model': self.config.get('embedding.default_model'),
                }

            # Create stats dictionary
            stats = {
                'doc_count': doc_count,
                'embedding_count': metadata.get('document_count', doc_count),
                'last_updated': metadata.get('created', 'Unknown'),
                'embedding_model': metadata.get('embedding_model', 'Unknown'),
                'languages': language_counts
            }

            return stats
        except Exception as e:
            debug_print(self.config, f"Error getting workspace stats: {str(e)}")
            return None

    def delete_workspace(self, workspace):
        """
        Delete a workspace and all its data

        Args:
            workspace (str): Workspace to delete

        Returns:
            bool: Success flag
        """
        debug_print(self.config, f"Deleting workspace: {workspace}")

        try:
            # Delete data directory
            data_dir = os.path.join("data", workspace)
            if os.path.exists(data_dir):
                shutil.rmtree(data_dir)

            # Delete documents directory
            body_dir = os.path.join("body", workspace)
            if os.path.exists(body_dir):
                shutil.rmtree(body_dir)

            # Remove from cache
            if workspace in self.vector_stores:
                del self.vector_stores[workspace]

            print(f"Workspace '{workspace}' deleted")
            return True
        except Exception as e:
            print(f"Error deleting workspace: {str(e)}")
            return False

    def verify_vector_store(self, workspace):
        """
        Perform a comprehensive verification of the vector store

        Args:
            workspace (str): Workspace to verify
        """
        print(f"\nVerifying vector store for workspace: {workspace}")

        # Check directories and files
        data_dir = os.path.join("data", workspace)
        vector_dir = os.path.join(data_dir, "vector_store")
        documents_dir = os.path.join(data_dir, "documents")

        print(f"Vector store directory: {vector_dir}")
        print(f"Documents directory: {documents_dir}")

        # Count documents in storage
        doc_count = 0
        if os.path.exists(documents_dir):
            doc_files = [f for f in os.listdir(documents_dir) if f.endswith('.json')]
            doc_count = len(doc_files)

        print(f"Document count in storage: {doc_count}")

        # Check vector store files
        if os.path.exists(vector_dir):
            vs_files = os.listdir(vector_dir)
            print(f"Vector store files: {len(vs_files)}")

            for file in vs_files:
                path = os.path.join(vector_dir, file)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    print(f"  - {file}: {self._format_size(size)}")

                    # For JSON files, check basic structure
                    if file.endswith('.json'):
                        try:
                            with open(path, 'r') as f:
                                data = json.load(f)

                                if file == 'docstore.json' and isinstance(data, dict):
                                    if 'docstore/docs' in data:
                                        vs_doc_count = len(data['docstore/docs'])
                                        print(f"    Documents in docstore: {vs_doc_count}")

                                        if vs_doc_count == 0:
                                            print("    ERROR: Docstore is empty!")
                                        elif vs_doc_count != doc_count:
                                            print(
                                                f"    WARNING: Document count mismatch - Storage: {doc_count}, Vector store: {vs_doc_count}")
                        except Exception as e:
                            print(f"    Error reading {file}: {str(e)}")
        else:
            print("Vector store directory does not exist!")

        # Test query
        print("\nTesting vector store with query...")
        try:
            # Ensure vector store is loaded
            loaded = self.storage_manager.load_vector_store(workspace)
            if not loaded:
                print("Failed to load vector store")
            else:
                print("Vector store loaded successfully")

                # Try a simple query
                test_query = "test query"
                results = self.storage_manager.query_documents(workspace, test_query, k=3)

                if not results:
                    print("Query returned no results!")
                else:
                    print(f"Query returned {len(results)} results:")
                    for i, doc in enumerate(results):
                        score = doc.get('relevance_score', 'N/A')
                        source = doc.get('metadata', {}).get('source', 'unknown')
                        print(f"  {i + 1}. Source: {source}, Score: {score}")
        except Exception as e:
            print(f"Error testing vector store: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def migrate_vector_store(self, workspace: str) -> bool:
        """
        Migrate an existing vector store to use consistent naming

        Args:
            workspace (str): Workspace to migrate

        Returns:
            bool: Success flag
        """
        debug_print(self.config, f"Migrating vector store for workspace: {workspace}")

        vector_dir = os.path.join("data", workspace, "vector_store")
        if not os.path.exists(vector_dir):
            debug_print(self.config, f"No vector store directory for workspace: {workspace}")
            return False

        try:
            # Check for prefixed vector store files
            prefixed_files = [f for f in os.listdir(vector_dir) if f.endswith('__vector_store.json')]
            if not prefixed_files:
                debug_print(self.config, "No prefixed vector store files to migrate")
                return False

            # Create backup directory
            backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir)

            # Copy all files to backup
            for file in os.listdir(vector_dir):
                shutil.copy2(os.path.join(vector_dir, file), os.path.join(backup_dir, file))

            # Import necessary components
            from llama_index.core import StorageContext, load_index_from_storage
            from llama_index.core.vector_stores.simple import SimpleVectorStore

            # Load the existing index
            original_storage = StorageContext.from_defaults(persist_dir=vector_dir)
            original_index = load_index_from_storage(original_storage)

            # Get all documents from the original index
            all_docs = list(original_index.docstore.docs.values())

            # Create new storage with consistent naming
            vector_store = SimpleVectorStore(collection_name="vector_store")
            new_storage = StorageContext.from_defaults(vector_store=vector_store)

            # Create new index with these documents
            from llama_index.core import VectorStoreIndex
            new_index = VectorStoreIndex(all_docs, storage_context=new_storage)

            # Remove original files
            for file in prefixed_files:
                os.remove(os.path.join(vector_dir, file))

            # Persist new index
            new_storage.persist(persist_dir=vector_dir)

            debug_print(self.config, f"Successfully migrated vector store")
            return True

        except Exception as e:
            debug_print(self.config, f"Error migrating vector store: {str(e)}")
            return False

    def rebuild_vector_store(self, workspace):
        """
        Completely rebuild the vector store for a workspace from scratch

        Args:
            workspace (str): Workspace to rebuild
        """
        print(f"\nRebuilding vector store for workspace: {workspace}")

        # Get documents
        docs = self.storage_manager.get_documents(workspace)

        if not docs:
            print(f"No documents found in workspace '{workspace}'")
            return

        print(f"Found {len(docs)} documents")

        # Create vector store directory
        vector_dir = os.path.join("data", workspace, "vector_store")
        if os.path.exists(vector_dir):
            backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Backing up existing vector store to {backup_dir}")
            os.rename(vector_dir, backup_dir)

        ensure_dir(vector_dir)

        # Determine the vector store type
        vector_store_type = self.config.get('storage.vector_store')
        print(f"Using vector store type: {vector_store_type}")

        # Build vector store
        print("Building fresh vector store...")

        # Force reinitialize the connector for clean start
        if hasattr(self.storage_manager, 'connector'):
            # Clear any cached state in the connector
            self.storage_manager.connector = self.storage_manager.factory.get_vector_store_connector()
            print("Created fresh connector instance")

        # Build vector store from scratch
        success = self.storage_manager.create_vector_store(workspace)

        if success:
            print("Vector store rebuilt successfully")

            # Verify the rebuild
            self.verify_vector_store(workspace)
        else:
            print("Failed to rebuild vector store")

    def fix_common_issues(self, workspace):
        """
        Attempt to fix common vector store issues

        Args:
            workspace (str): Workspace to fix
        """
        print(f"\nAttempting to fix vector store issues for workspace: {workspace}")

        # Check for corrupted or empty files
        vector_dir = os.path.join("data", workspace, "vector_store")
        if not os.path.exists(vector_dir):
            print("Vector store directory doesn't exist - nothing to fix")
            return False

        # Check if files exist but are invalid
        docstore_path = os.path.join(vector_dir, "docstore.json")
        index_store_path = os.path.join(vector_dir, "index_store.json")
        vector_store_path = os.path.join(vector_dir, "vector_store.json")

        files_valid = True

        for path in [docstore_path, index_store_path, vector_store_path]:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                        # File exists and is valid JSON
                except Exception:
                    print(f"File {os.path.basename(path)} is corrupted")
                    files_valid = False
            else:
                print(f"File {os.path.basename(path)} is missing")
                files_valid = False

        if not files_valid:
            print("Vector store files are corrupted or missing - rebuilding from scratch")
            self.rebuild_vector_store(workspace)
            return True

        # Check if docstore has documents
        try:
            with open(docstore_path, 'r') as f:
                data = json.load(f)
                if 'docstore/docs' in data and len(data['docstore/docs']) == 0:
                    print("Docstore is empty - rebuilding from scratch")
                    self.rebuild_vector_store(workspace)
                    return True
        except Exception:
            # Already handled above
            pass

        # If we get here, no obvious issues were found
        print("No obvious issues found with vector store files")

        # Still, let's verify it works correctly
        self.verify_vector_store(workspace)

        return False


def normalize_source_path(metadata):
        """
        Normalize source path in metadata for consistent handling.

        Args:
            metadata (dict): The metadata dictionary to normalize

        Returns:
            dict: Normalized metadata
        """
        # If metadata is None, return empty dict
        if metadata is None:
            return {}

        # Make a copy to avoid modifying the original
        result = metadata.copy()

        # Ensure source paths are consistent
        if 'source' in result:
            # Convert Windows paths to Unix format for consistency
            source = result['source'].replace('\\', '/')

            # Remove any leading ./ or / for consistency
            while source.startswith('./'):
                source = source[2:]
            while source.startswith('/'):
                source = source[1:]

            result['source'] = source

        return result

class VectorStoreInspector:
    """Provides utilities to inspect and diagnose vector store content"""

    def __init__(self, config, storage_manager=None):
        """
        Initialize the vector store inspector

        Args:
            config: Configuration object
            storage_manager: Storage manager instance
        """
        from core.engine.storage import StorageManager

        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        debug_print(config, "Vector store inspector initialized")

    """
    Vector store inspection utilities
    """

    def inspect_workspace(self, workspace):
        """
        Inspect a workspace's vector store and data

        Args:
            workspace (str): Workspace to inspect
        """
        debug_print(self.config, f"Inspecting workspace: {workspace}")

        # Check for workspace directories
        data_dir = os.path.join("data", workspace)
        body_dir = os.path.join("body", workspace)

        print(f"\nInspecting workspace: {workspace}")

        # Check workspace directories
        print("\nWorkspace Directories:")
        print(f"  Data Directory: {data_dir} - {'Exists' if os.path.exists(data_dir) else 'Missing'}")
        print(f"  Body Directory: {body_dir} - {'Exists' if os.path.exists(body_dir) else 'Missing'}")

        # Check document files
        documents_dir = os.path.join(data_dir, "documents")
        if os.path.exists(documents_dir):
            document_files = [f for f in os.listdir(documents_dir) if f.endswith('.json')]
            print(f"\nDocument JSON Files: {len(document_files)}")
            for i, file in enumerate(document_files[:5]):  # Show first 5
                print(f"  {i + 1}. {file}")

                # Peek inside the document file
                try:
                    with open(os.path.join(documents_dir, file), 'r', encoding='utf-8') as f:
                        doc = json.load(f)
                        source = doc.get('metadata', {}).get('source', 'unknown')
                        print(f"     Source: {source}")
                        print(f"     Content length: {len(doc.get('content', ''))}")
                        print(f"     Has processed content: {'Yes' if doc.get('processed_content') else 'No'}")
                except Exception as e:
                    print(f"     Error reading file: {str(e)}")

            if len(document_files) > 5:
                print(f"  ... and {len(document_files) - 5} more")
        else:
            print("\nNo document directory found")

        # Check vector store type and files
        vector_store_type = self.config.get('storage.vector_store')
        vector_dir = os.path.join(data_dir, "vector_store")

        print(f"\nVector Store Type: {vector_store_type}")
        print(f"Vector Store Directory: {vector_dir} - {'Exists' if os.path.exists(vector_dir) else 'Missing'}")

        # If vector store directory exists, list and check all files
        if os.path.exists(vector_dir):
            print("\nVector Store Files:")
            for file in os.listdir(vector_dir):
                file_path = os.path.join(vector_dir, file)
                size = os.path.getsize(file_path) if os.path.isfile(file_path) else 0
                print(f"  {file}: {self._format_size(size)}")

                # For JSON files, validate and show basic structure
                if file.endswith('.json') and os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                print(f"    Keys: {list(data.keys())[:5]}{' ...' if len(data.keys()) > 5 else ''}")

                                # Count documents for docstore
                                if file == 'docstore.json' and 'docstore/docs' in data:
                                    print(f"    Documents in docstore: {len(data['docstore/docs'])}")

                                    # Show a sample document
                                    if data['docstore/docs']:
                                        doc_id = next(iter(data['docstore/docs']))
                                        doc = data['docstore/docs'][doc_id]
                                        print(f"    Sample document ID: {doc_id}")
                                        if 'metadata' in doc:
                                            print(
                                                f"    Sample metadata: {doc.get('metadata', {}).get('source', 'unknown')}")
                            else:
                                print(f"    JSON data type: {type(data).__name__}")
                    except Exception as e:
                        print(f"    Error reading JSON: {str(e)}")

        # Attempt to load the vector store and perform a test query
        print("\nAttempting to load vector store and perform test query...")
        try:
            loaded = self.storage_manager.load_vector_store(workspace)
            print(f"  Load vector store: {'Success' if loaded else 'Failed'}")

            if loaded:
                # Get a simple test query
                test_query = "test"
                results = self.storage_manager.query_documents(workspace, test_query, k=1)
                print(f"  Test query results: {len(results)} documents retrieved")

                if results:
                    doc = results[0]
                    print(f"    First result source: {doc.get('metadata', {}).get('source', 'unknown')}")
                    print(f"    First result score: {doc.get('relevance_score', 'N/A')}")
                else:
                    print("    No documents retrieved from test query")
        except Exception as e:
            print(f"  Error testing vector store: {str(e)}")
            import traceback
            print(f"  {traceback.format_exc()}")

    def _inspect_llama_index_store(self, workspace):
        """Inspect LlamaIndex vector store"""
        vector_dir = os.path.join("data", workspace, "vector_store")

        if not os.path.exists(vector_dir):
            print("  No LlamaIndex vector store directory found")
            return

        print("  LlamaIndex Files:")
        expected_files = ["docstore.json", "index_store.json", "vector_store.json"]

        for file in expected_files:
            path = os.path.join(vector_dir, file)
            status = "Found" if os.path.exists(path) else "Missing"
            print(f"    {file}: {status}")

            if status == "Found":
                size = os.path.getsize(path)
                print(f"      Size: {self._format_size(size)}")

                # Basic JSON validation
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    print(f"      Valid JSON: Yes")

                    # For docstore, count documents
                    if file == "docstore.json" and "docstore/docs" in data:
                        doc_count = len(data["docstore/docs"])
                        print(f"      Documents: {doc_count}")
                except:
                    print(f"      Valid JSON: No (corrupted)")

    def _inspect_haystack_store(self, workspace):
        """Inspect Haystack document store"""
        store_dir = os.path.join("data", workspace, "document_store")

        if not os.path.exists(store_dir):
            print("  No Haystack document store directory found")
            return

        print("  Haystack Files:")

        files = os.listdir(store_dir)
        for file in files:
            path = os.path.join(store_dir, file)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"    {file}: {self._format_size(size)}")

    def _inspect_default_store(self, workspace):
        """Inspect default vector store"""
        vector_dir = os.path.join("data", workspace, "vector_store")

        if not os.path.exists(vector_dir):
            print("  No vector store directory found")
            return

        print("  Vector Store Files:")

        files = os.listdir(vector_dir)
        for file in files:
            path = os.path.join(vector_dir, file)
            if os.path.isfile(path):
                size = os.path.getsize(path)
                print(f"    {file}: {self._format_size(size)}")

    def _check_embedding_status(self, workspace, docs):
        """Check embedding status for documents"""
        vector_store_type = self.config.get('storage.vector_store')
        vector_dir = os.path.join("data", workspace, "vector_store")

        if not os.path.exists(vector_dir):
            print("\nNo vector store directory found - documents may not be embedded")
            return

        # Try to determine if documents are embedded
        if vector_store_type == 'llama_index':
            docstore_path = os.path.join(vector_dir, "docstore.json")
            if os.path.exists(docstore_path):
                try:
                    with open(docstore_path, 'r') as f:
                        data = json.load(f)

                    if "docstore/docs" in data:
                        embedded_count = len(data["docstore/docs"])
                        print(f"\nEmbedded Documents: {embedded_count} / {len(docs)}")

                        if embedded_count < len(docs):
                            print("Warning: Not all documents are embedded")
                except:
                    print("\nCould not verify embedding status (corrupted docstore.json)")

    def dump_document_content(self, workspace, limit=5, format="text"):
        """
        Dump the content of documents in a workspace

        Args:
            workspace (str): Workspace to inspect
            limit (int): Maximum number of documents to dump
            format (str): Output format ('text', 'json')
        """
        debug_print(self.config, f"Dumping document content for workspace: {workspace}")

        # Get documents
        docs = self.storage_manager.get_documents(workspace)

        if not docs:
            print(f"No documents found in workspace '{workspace}'")
            return

        print(f"\nDumping document content for workspace: {workspace}")
        print(f"Total documents: {len(docs)}")

        # Limit number of docs
        docs_to_dump = docs[:limit]

        if format == "json":
            # Create a clean version for JSON output
            clean_docs = []
            for doc in docs_to_dump:
                clean_doc = {
                    "metadata": doc.get("metadata", {}),
                    "content_preview": doc.get("content", "")[:200] + "..." if len(
                        doc.get("content", "")) > 200 else doc.get("content", ""),
                    "processed_content_preview": doc.get("processed_content", "")[:200] + "..." if len(
                        doc.get("processed_content", "")) > 200 else doc.get("processed_content", "")
                }
                clean_docs.append(clean_doc)

            print(json.dumps(clean_docs, indent=2))
        else:
            # Text format
            for i, doc in enumerate(docs_to_dump):
                print(f"\n--- Document {i + 1} ---")
                print(f"Source: {doc.get('metadata', {}).get('source', 'unknown')}")
                print(f"Language: {doc.get('metadata', {}).get('language', 'unknown')}")

                print("\nContent Preview:")
                content = doc.get("content", "")
                print(content[:500] + "..." if len(content) > 500 else content)

                print("\nProcessed Content Preview:")
                processed = doc.get("processed_content", "")
                print(processed[:500] + "..." if len(processed) > 500 else processed)

        if len(docs) > limit:
            print(f"\n... and {len(docs) - limit} more documents")

    def _dump_llama_index_metadata(self, workspace):
        """Dump LlamaIndex metadata"""
        vector_dir = os.path.join("data", workspace, "vector_store")

        # Check index store
        index_store_path = os.path.join(vector_dir, "index_store.json")
        if os.path.exists(index_store_path):
            print("\nIndex Store:")
            try:
                with open(index_store_path, 'r') as f:
                    data = json.load(f)

                # Print summary instead of full dump
                print(f"  Number of indices: {len(data.get('indices', {}))}")

                for index_id, index_data in data.get('indices', {}).items():
                    print(f"  Index ID: {index_id}")
                    print(f"    Type: {index_data.get('index_id', 'unknown')}")
                    print(f"    Summary: {index_data.get('summary', 'No summary')}")
            except:
                print("  Error: Could not parse index store")

        # Check document store
        docstore_path = os.path.join(vector_dir, "docstore.json")
        if os.path.exists(docstore_path):
            print("\nDocument Store:")
            try:
                with open(docstore_path, 'r') as f:
                    data = json.load(f)

                docs = data.get("docstore/docs", {})
                print(f"  Number of documents: {len(docs)}")

                # Sample first few documents
                for i, (doc_id, doc) in enumerate(list(docs.items())[:3]):
                    print(f"\n  Document {i + 1} (ID: {doc_id}):")

                    if "text" in doc:
                        text = doc["text"]
                        print(f"    Text: {text[:100]}..." if len(text) > 100 else f"    Text: {text}")

                    if "embedding" in doc:
                        embedding = doc.get("embedding")
                        if embedding:
                            print(f"    Embedding: [vector with {len(embedding)} dimensions]")
                        else:
                            print("    Embedding: None")

                if len(docs) > 3:
                    print(f"\n  ... and {len(docs) - 3} more documents")
            except Exception as e:
                print(f"  Error: Could not parse document store: {str(e)}")

    def _dump_haystack_metadata(self, workspace):
        """Dump Haystack metadata"""
        store_dir = os.path.join("data", workspace, "document_store")

        if not os.path.exists(store_dir):
            print("  No Haystack document store found")
            return

        # Find metadata files
        metadata_files = [f for f in os.listdir(store_dir) if f.endswith('.json')]

        for file in metadata_files:
            path = os.path.join(store_dir, file)
            print(f"\nFile: {file}")

            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                # Print summary
                print(f"  Content type: {type(data).__name__}")

                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())}")

                    if "documents" in data:
                        docs = data["documents"]
                        print(f"  Documents: {len(docs)}")

                        for i, doc in enumerate(docs[:3]):
                            print(f"\n  Document {i + 1}:")
                            print(f"    ID: {doc.get('id', 'unknown')}")

                            content = doc.get("content", "")
                            print(
                                f"    Content: {content[:100]}..." if len(content) > 100 else f"    Content: {content}")

                            if "embedding" in doc:
                                embedding = doc.get("embedding")
                                if embedding:
                                    print(f"    Embedding: [vector with {len(embedding)} dimensions]")
                                else:
                                    print("    Embedding: None")

                        if len(docs) > 3:
                            print(f"\n  ... and {len(docs) - 3} more documents")
            except:
                print("  Error: Could not parse file")

    def _dump_default_metadata(self, workspace):
        """Dump default metadata"""
        vector_dir = os.path.join("data", workspace, "vector_store")

        if not os.path.exists(vector_dir):
            print("  No vector store directory found")
            return

        # Find metadata files
        metadata_files = [f for f in os.listdir(vector_dir) if f.endswith('.json')]

        for file in metadata_files:
            path = os.path.join(vector_dir, file)
            print(f"\nFile: {file}")

            try:
                with open(path, 'r') as f:
                    data = json.load(f)

                # Print summary
                if isinstance(data, dict):
                    top_level_keys = list(data.keys())
                    print(f"  Top-level keys: {top_level_keys}")

                    # If less than 10 keys, show content of each key
                    if len(top_level_keys) < 10:
                        for key in top_level_keys:
                            value = data[key]
                            if isinstance(value, dict):
                                print(f"  {key}: {{{len(value)} items}}")
                            elif isinstance(value, list):
                                print(f"  {key}: [{len(value)} items]")
                            else:
                                print(f"  {key}: {value}")
            except:
                print("  Error: Could not parse file")

    def test_query_pipeline(self, workspace, test_query="test"):
        """
        Test the query pipeline for a workspace

        Args:
            workspace (str): Workspace to test
            test_query (str): Test query to run
        """
        debug_print(self.config, f"Testing query pipeline for workspace: {workspace}")

        print(f"\nTesting query pipeline for workspace: {workspace}")
        print(f"Test query: '{test_query}'")

        # Check if workspace exists
        data_dir = os.path.join("data", workspace)
        if not os.path.exists(data_dir):
            print(f"Error: Workspace '{workspace}' does not exist")
            return

        # Test each step of the query pipeline
        print("\nStep 1: Load vector store")
        loaded = self.storage_manager.load_vector_store(workspace)
        print(f"  Result: {'Success' if loaded else 'Failed'}")

        if not loaded:
            print("  Cannot proceed - failed to load vector store")
            return

        print("\nStep 2: Preprocess query")
        try:
            from core.language.processor import TextProcessor
            processor = TextProcessor(self.config)
            processed_query = processor.preprocess(test_query)
            print(f"  Original: '{test_query}'")
            print(f"  Processed: '{processed_query['processed']}'")
            print(f"  Language: {processed_query['language']}")
        except Exception as e:
            print(f"  Error: {str(e)}")
            return

        print("\nStep 3: Query vector store")
        try:
            docs = self.storage_manager.query_documents(workspace, processed_query['processed'], k=3)
            print(f"  Retrieved documents: {len(docs)}")

            if docs:
                print("\nRetrieved Documents:")
                for i, doc in enumerate(docs):
                    print(f"\n  Document {i + 1}:")
                    print(f"    Source: {doc.get('metadata', {}).get('source', 'unknown')}")

                    if 'relevance_score' in doc:
                        print(f"    Score: {doc['relevance_score']}")

                    content = doc.get("content", "")
                    print(f"    Content: {content[:100]}..." if len(content) > 100 else f"    Content: {content}")
            else:
                print("\n  No documents retrieved. Possible issues:")
                print("    - Empty vector store")
                print("    - Documents not properly embedded")
                print("    - Query terms not found in documents")
                print("    - Similarity threshold too high")
        except Exception as e:
            print(f"  Error: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def rebuild_vector_store(self, workspace):
        """
        Completely rebuild the vector store for a workspace from scratch

        Args:
            workspace (str): Workspace to rebuild
        """
        print(f"\nRebuilding vector store for workspace: {workspace}")

        # Get documents
        docs = self.storage_manager.get_documents(workspace)

        if not docs:
            print(f"No documents found in workspace '{workspace}'")
            return

        print(f"Found {len(docs)} documents")

        # Create vector store directory
        vector_dir = os.path.join("data", workspace, "vector_store")
        if os.path.exists(vector_dir):
            backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Backing up existing vector store to {backup_dir}")

            # Create backup directory
            if not os.path.exists(backup_dir):
                os.makedirs(backup_dir)

            # Copy files to backup instead of renaming
            import shutil
            for file in os.listdir(vector_dir):
                src_path = os.path.join(vector_dir, file)
                dst_path = os.path.join(backup_dir, file)
                if os.path.isfile(src_path):
                    shutil.copy2(src_path, dst_path)

            # Remove the original directory
            shutil.rmtree(vector_dir)

        # Create new empty directory
        ensure_dir(vector_dir)

        # Determine the vector store type
        vector_store_type = self.config.get('storage.vector_store')
        print(f"Using vector store type: {vector_store_type}")

        # Build vector store
        print("Building fresh vector store...")

        # Force reinitialize the connector for clean start
        if hasattr(self.storage_manager, 'connector'):
            # Clear any cached state in the connector
            self.storage_manager.connector = self.storage_manager.factory.get_vector_store_connector()
            print("Created fresh connector instance")

        # Build vector store from scratch
        success = self.storage_manager.create_vector_store(workspace)

        if success:
            print("Vector store rebuilt successfully")

            # Verify the rebuild
            self.verify_vector_store(workspace)
        else:
            print("Failed to rebuild vector store")

        return success

    def _format_size(self, size_bytes):
        """Format file size in a human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

    def migrate_vector_store(self, workspace: str) -> bool:
        """
        Migrate an existing vector store to use consistent naming

        Args:
            workspace (str): Workspace to migrate

        Returns:
            bool: Success flag
        """
        print(f"Migrating vector store for workspace: {workspace}")

        vector_dir = os.path.join("data", workspace, "vector_store")
        if not os.path.exists(vector_dir):
            print(f"No vector store directory for workspace: {workspace}")
            return False

        try:
            # Check for any prefixed vector store files (e.g. default__vector_store.json, image__vector_store.json)
            prefixed_files = [f for f in os.listdir(vector_dir)
                              if f.endswith('__vector_store.json') or
                              f.endswith('_vector_store.json')]

            if not prefixed_files:
                print("No prefixed vector store files to migrate")
                return False

            print(f"Found {len(prefixed_files)} prefixed vector store files: {prefixed_files}")

            # Create backup directory
            backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(backup_dir)
            print(f"Created backup directory: {backup_dir}")

            # Copy all files to backup
            for file in os.listdir(vector_dir):
                shutil.copy2(os.path.join(vector_dir, file), os.path.join(backup_dir, file))
            print("Backed up all vector store files")

            try:
                # Rename the most recent or largest prefixed file to vector_store.json
                largest_file = None
                largest_size = 0

                for file in prefixed_files:
                    file_path = os.path.join(vector_dir, file)
                    file_size = os.path.getsize(file_path)
                    if file_size > largest_size:
                        largest_size = file_size
                        largest_file = file

                if largest_file:
                    print(
                        f"Using largest file as primary vector store: {largest_file} ({self._format_size(largest_size)})")
                    shutil.copy2(
                        os.path.join(vector_dir, largest_file),
                        os.path.join(vector_dir, "vector_store.json")
                    )
                    print("Created vector_store.json")

                # Check if we now have vector_store.json
                if os.path.exists(os.path.join(vector_dir, "vector_store.json")):
                    print("Migration successful: vector_store.json created")
                    print("You may need to rebuild the vector store if queries still return no results")
                    return True
                else:
                    print("Failed to create vector_store.json")
                    return False

            except Exception as e:
                print(f"Error during file migration: {str(e)}")
                import traceback
                print(traceback.format_exc())

                # Fall back to rebuilding the vector store
                print("Attempting to rebuild vector store from scratch...")
                self.rebuild_vector_store(workspace)
                return True

        except Exception as e:
            print(f"Error migrating vector store: {str(e)}")
            return False

    def _get_total_document_count(self, workspace):
        """
        Get the total number of documents in the vector store

        Args:
            workspace (str): Target workspace

        Returns:
            int: Total number of documents
        """
        try:
            # Use StorageManager to get documents
            docs = self.storage_manager.get_documents(workspace)
            return len(docs)
        except Exception as e:
            debug_print(self.config, f"Error getting document count: {str(e)}")
            return 0

    def _dump_llama_index_details(self, workspace):
        """
        Dump detailed LlamaIndex vector store information

        Args:
            workspace (str): Target workspace
        """
        vector_dir = os.path.join("data", workspace, "vector_store")

        # Paths for key files
        index_store_path = os.path.join(vector_dir, "index_store.json")
        docstore_path = os.path.join(vector_dir, "docstore.json")
        vector_store_path = os.path.join(vector_dir, "vector_store.json")

        # Track document count and details
        doc_count = 0
        embedding_dim = None
        language_distribution = {}

        try:
            # Attempt to load documents via StorageManager
            docs = self.storage_manager.get_documents(workspace)
            doc_count = len(docs)

            # Analyze language distribution
            for doc in docs:
                lang = doc.get('metadata', {}).get('language', 'unknown')
                language_distribution[lang] = language_distribution.get(lang, 0) + 1

            print("\nDocument Overview:")
            print(f"  Total Documents: {doc_count}")

            # Language Distribution
            print("\n  Language Distribution:")
            for lang, count in language_distribution.items():
                percentage = (count / doc_count) * 100 if doc_count > 0 else 0
                print(f"    {lang}: {count} documents ({percentage:.1f}%)")

            # Index Store Analysis
            if os.path.exists(index_store_path):
                print("\nIndex Store Details:")
                try:
                    with open(index_store_path, 'r') as f:
                        index_data = json.load(f)

                    indices = index_data.get('indices', {})
                    print(f"  Number of Indices: {len(indices)}")

                    if indices:
                        for idx, details in indices.items():
                            print(f"  Index {idx}:")
                            print(f"    Type: {details.get('type', 'Unknown')}")
                            print(f"    Summary: {details.get('summary', 'No summary')}")
                except Exception as e:
                    print(f"  Error reading index store: {str(e)}")

            # Document Store Analysis
            if os.path.exists(docstore_path):
                print("\nDocument Store Details:")
                try:
                    with open(docstore_path, 'r') as f:
                        docstore_data = json.load(f)

                    docs = docstore_data.get("docstore/docs", {})
                    print(f"  Stored Documents: {len(docs)}")

                    # Sample document metadata
                    if docs:
                        sample_docs = list(docs.items())[:3]
                        print("  Sample Documents:")
                        for doc_id, doc_data in sample_docs:
                            print(f"    Document ID: {doc_id}")

                            # Check for embedding
                            if "embedding" in doc_data:
                                embedding = doc_data["embedding"]
                                embedding_dim = len(embedding)
                                print(f"    Embedding Dimensions: {embedding_dim}")
                except Exception as e:
                    print(f"  Error reading document store: {str(e)}")

            # Vector Store Analysis
            if os.path.exists(vector_store_path):
                print("\nVector Store Details:")
                try:
                    with open(vector_store_path, 'r') as f:
                        vector_data = json.load(f)

                    print("  Top-level Keys:")
                    for key, value in vector_data.items():
                        if isinstance(value, (list, dict)):
                            print(f"    {key}: {len(value)} items")
                        else:
                            print(f"    {key}: {value}")
                except Exception as e:
                    print(f"  Error reading vector store: {str(e)}")

            # Additional diagnostic information
            print("\nVector Store File Analysis:")
            for file in ['index_store.json', 'docstore.json', 'vector_store.json']:
                filepath = os.path.join(vector_dir, file)
                if os.path.exists(filepath):
                    size = os.path.getsize(filepath)
                    print(f"  {file}: {self._format_size(size)}")
                else:
                    print(f"  {file}: Not found")

        except Exception as e:
            print(f"Comprehensive error during vector store inspection: {str(e)}")
            import traceback
            traceback.print_exc()

        # Final summary
        print("\n=== Vector Store Summary ===")
        print(f"Total Documents: {doc_count}")
        if embedding_dim:
            print(f"Embedding Dimensions: {embedding_dim}")
        print("==========================")

    def dump_vector_store(self, workspace):
        """
        Dump vector store metadata for a workspace

        Args:
            workspace (str): Workspace to inspect
        """
        debug_print(self.config, f"Dumping vector store for workspace: {workspace}")

        vector_store_type = self.config.get('storage.vector_store')

        # First, try to load the vector store
        loaded = self.storage_manager.load_vector_store(workspace)
        if not loaded:
            print(f"Failed to load vector store for workspace '{workspace}'")
            return

        print(f"\nDumping vector store for workspace: {workspace}")
        print(f"Vector store type: {vector_store_type}")

        # Direct query to get document count
        try:
            results = self.storage_manager.query_documents(workspace, "test query", k=1)

            print("\nVector Store Statistics:")

            # Attempt to get total document count
            total_docs = self._get_total_document_count(workspace)
            print(f"  Total Documents: {total_docs}")

            # Get vector store directory details
            vector_dir = os.path.join("data", workspace, "vector_store")
            if os.path.exists(vector_dir):
                print("\nVector Store Files:")
                for file in os.listdir(vector_dir):
                    filepath = os.path.join(vector_dir, file)
                    if os.path.isfile(filepath):
                        size = os.path.getsize(filepath)
                        print(f"  {file}: {self._format_size(size)}")

            # Check Index Store for additional details
            if vector_store_type == 'llama_index':
                self._dump_llama_index_details(workspace)

        except Exception as e:
            print(f"Error retrieving vector store details: {str(e)}")

    def verify_vector_store(self, workspace):
        """
        Perform a comprehensive verification of the vector store

        Args:
            workspace (str): Workspace to verify
        """
        print(f"\nVerifying vector store for workspace: {workspace}")

        # Check directories and files
        data_dir = os.path.join("data", workspace)
        vector_dir = os.path.join(data_dir, "vector_store")
        documents_dir = os.path.join(data_dir, "documents")

        print(f"Vector store directory: {vector_dir}")
        print(f"Documents directory: {documents_dir}")

        # Count documents in storage
        doc_count = 0
        if os.path.exists(documents_dir):
            doc_files = [f for f in os.listdir(documents_dir) if f.endswith('.json')]
            doc_count = len(doc_files)

        print(f"Document count in storage: {doc_count}")

        # Check vector store files
        if os.path.exists(vector_dir):
            vs_files = os.listdir(vector_dir)
            print(f"Vector store files: {len(vs_files)}")

            for file in vs_files:
                path = os.path.join(vector_dir, file)
                if os.path.isfile(path):
                    size = os.path.getsize(path)
                    print(f"  - {file}: {self._format_size(size)}")

                    # For JSON files, check basic structure
                    if file.endswith('.json'):
                        try:
                            with open(path, 'r') as f:
                                data = json.load(f)

                                if file == 'docstore.json' and isinstance(data, dict):
                                    if 'docstore/docs' in data:
                                        vs_doc_count = len(data['docstore/docs'])
                                        print(f"    Documents in docstore: {vs_doc_count}")

                                        if vs_doc_count == 0:
                                            print("    ERROR: Docstore is empty!")
                                        elif vs_doc_count != doc_count:
                                            print(
                                                f"    WARNING: Document count mismatch - Storage: {doc_count}, Vector store: {vs_doc_count}")
                        except Exception as e:
                            print(f"    Error reading {file}: {str(e)}")
        else:
            print("Vector store directory does not exist!")

        # Test query
        print("\nTesting vector store with query...")
        try:
            # Ensure vector store is loaded
            loaded = self.storage_manager.load_vector_store(workspace)
            if not loaded:
                print("Failed to load vector store")
            else:
                print("Vector store loaded successfully")

                # Try a simple query
                test_query = "test query"
                results = self.storage_manager.query_documents(workspace, test_query, k=3)

                if not results:
                    print("Query returned no results!")
                else:
                    print(f"Query returned {len(results)} results:")
                    for i, doc in enumerate(results):
                        score = doc.get('relevance_score', 'N/A')
                        source = doc.get('metadata', {}).get('source', 'unknown')
                        print(f"  {i + 1}. Source: {source}, Score: {score}")
        except Exception as e:
            print(f"Error testing vector store: {str(e)}")
            import traceback
            print(traceback.format_exc())