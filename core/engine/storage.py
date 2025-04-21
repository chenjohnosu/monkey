"""
Vector store management using LlamaIndex and Haystack
Refactored to reduce duplicate code and improve consistency
"""

import os
import json
import hashlib
import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

from core.engine.utils import ensure_dir, save_json, load_json, create_timestamped_backup, format_size
from core.engine.logging import debug, info, error, warning, debug
from core.engine.common import safe_execute, get_workspace_dirs, ensure_workspace_dirs
from core.connectors.connector_factory import ConnectorFactory


class StorageManager:
    """Manages document storage and vector databases using LlamaIndex or Haystack"""

    def __init__(self, config):
        """Initialize the storage manager"""
        self.config = config
        self.vector_stores = {}  # Cache for loaded vector stores
        self.factory = ConnectorFactory(config)
        self.connector = self.factory.get_vector_store_connector()
        debug(config, f"Storage manager initialized with connector type: {self.config.get('storage.vector_store')}")

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
        debug(self.config, f"Adding document to workspace: {workspace}, source: {source_path}")

        # Use get_workspace_dirs from common.py to get standard directory paths
        dirs = get_workspace_dirs(workspace)
        doc_dir = dirs['documents']
        ensure_dir(doc_dir)

        # Create safe filename from source path
        filename = hashlib.md5(source_path.encode('utf-8')).hexdigest() + '.json'
        filepath = Path(doc_dir) / filename

        # Create document object
        document = {
            'content': content,
            'processed_content': processed_content,
            'metadata': metadata
        }

        # Save document to file using save_json from utils.py
        save_json(filepath, document)

        # Add to vector store
        debug(self.config, f"Adding document to vector store: {source_path}")
        return safe_execute(
            self._add_to_vector_store,
            workspace, document,
            error_message="Error adding document to vector store"
        )

    def remove_document(self, workspace, source_path):
        """
        Remove a document from storage

        Args:
            workspace (str): Target workspace
            source_path (str): Document source path

        Returns:
            bool: Success flag
        """
        debug(self.config, f"Removing document from workspace: {workspace}, source: {source_path}")

        # Create safe filename from source path
        filename = hashlib.md5(source_path.encode('utf-8')).hexdigest() + '.json'
        dirs = get_workspace_dirs(workspace)
        filepath = Path(dirs['documents']) / filename

        # Remove file if it exists
        if filepath.exists():
            try:
                filepath.unlink()
                debug(self.config, f"Removed document file: {filepath}")
                return True
            except Exception as e:
                debug(self.config, f"Error removing document file: {str(e)}")
                return False
        else:
            debug(self.config, f"Document file not found: {filepath}")
            return False

    def _add_to_vector_store(self, workspace, document):
        """
        Add a document to the vector store

        Args:
            workspace (str): Target workspace
            document (dict): Document object
        """
        vector_store_type = self.config.get('storage.vector_store')
        debug(self.config, f"Adding document to {vector_store_type} vector store")

        # Create list with single document
        documents = [document]

        # Use the appropriate connector
        try:
            success = self.connector.add_documents(workspace, documents)
            if not success:
                debug(self.config, "Failed to add document to vector store")
            return success
        except Exception as e:
            debug(self.config, f"Error adding document to vector store: {str(e)}")
            import traceback
            debug(self.config, traceback.format_exc())
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
        debug(self.config, f"Updating vector store for workspace: {workspace}")

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

            # Use appropriate connector update method
            return safe_execute(
                self.connector.add_documents,
                workspace, docs_to_update,
                error_message=f"Error updating vector store for workspace '{workspace}'"
            )

        except Exception as e:
            error(f"Error updating vector store: {str(e)}")
            import traceback
            debug(traceback.format_exc())
            return False

    def get_documents(self, workspace):
        """
        Get all documents in a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            list: Document objects
        """
        debug(self.config, f"Getting documents from workspace: {workspace}")

        # Get standard directory paths using common.py function
        dirs = get_workspace_dirs(workspace)
        workspace_dir = Path(dirs['data'])
        documents_dir = Path(dirs['documents'])

        # Detailed diagnostics
        debug("Workspace Document Retrieval:")
        debug(f"  Workspace Directory: {workspace_dir}")
        debug(f"  Documents Directory: {documents_dir}")

        # Check directory existence
        if not workspace_dir.exists():
            error(f"ERROR: Workspace directory does not exist: {workspace_dir}")
            return []

        if not documents_dir.exists():
            error(f"ERROR: Documents directory does not exist: {documents_dir}")
            # Attempt to create the directory
            ensure_dir(documents_dir)
            info(f"Created documents directory: {documents_dir}")
            return []

        documents = []

        # List files in the documents directory
        try:
            document_files = [f.name for f in documents_dir.glob('*.json')]
            print(f"  JSON Files in Documents Directory: {len(document_files)}")

            # If no files, provide more context
            if not document_files:
                warning("  Warning: No JSON document files found")
                # List all files in the directory to understand why
                all_files = [f.name for f in documents_dir.iterdir()]
                if all_files:
                    print("  All files in documents directory:")
                    for file in all_files:
                        print(f"    - {file}")
                else:
                    warning("  No files found in documents directory")
        except Exception as e:
            print(f"  Error listing document files: {str(e)}")
            return []

        # Load documents from files using load_json from utils.py
        for filename in document_files:
            filepath = documents_dir / filename
            document = load_json(filepath)
            if document:
                documents.append(document)

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
        debug(self.config, f"Querying workspace '{workspace}' with k={k}")

        # Use the appropriate connector
        return safe_execute(
            self._execute_query,
            workspace, query, k,
            error_message=f"Error querying vector store for workspace '{workspace}'",
            default_return=[]
        )

    def _execute_query(self, workspace, query, k):
        """Execute query using connector and normalize results"""
        results = self.connector.query(workspace, query, k)

        # Normalize metadata in results
        for doc in results:
            if 'metadata' in doc:
                doc['metadata'] = normalize_source_path(doc['metadata'])

        return results

    def get_processed_files(self, workspace):
        """
        Get a dictionary of processed files and their metadata

        Args:
            workspace (str): Target workspace

        Returns:
            dict: Dictionary mapping file paths to metadata
        """
        debug(self.config, f"Getting processed files for workspace: {workspace}")

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
        debug(self.config, f"Creating vector store for workspace: {workspace}")

        # Get all documents
        documents = self.get_documents(workspace)
        if not documents:
            print(f"No documents found in workspace '{workspace}'")
            return False

        print(f"Building vector store for {len(documents)} source files")

        # Get vector store directory from common.py function
        dirs = get_workspace_dirs(workspace)
        vector_dir = Path(dirs['vector_store'])

        if vector_dir.exists():
            # Create backup using utils.py function
            backup_dir = create_timestamped_backup(vector_dir)
            if backup_dir:
                debug(self.config, f"Backed up existing vector store to {backup_dir}")

            # Remove existing directory and recreate
            import shutil
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
                    docstore_path = vector_dir / "docstore.json"
                    if docstore_path.exists():
                        try:
                            docstore_data = load_json(docstore_path)
                            if docstore_data and 'docstore/docs' in docstore_data:
                                index_doc_count = len(docstore_data['docstore/docs'])
                                if index_doc_count != len(documents):
                                    print(f"Note: {len(documents)} source files resulted in {index_doc_count} vector index entries")
                        except Exception:
                            pass

                print(f"Vector store created successfully with {len(documents)} source files")

                # Save vector store metadata
                metadata_path = vector_dir / "metadata.json"
                save_json(metadata_path, {
                    "created": datetime.datetime.now().isoformat(),
                    "source_file_count": len(documents),
                    "embedding_model": self.config.get('embedding.default_model')
                })

            return success

        except Exception as e:
            error(f"Error creating vector store: {str(e)}")
            import traceback
            debug(traceback.format_exc())
            return False

    def load_vector_store(self, workspace):
        """
        Load a vector store for a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            bool: Success flag
        """
        debug(self.config, f"Loading vector store for workspace: {workspace}")

        # Check if already loaded
        if workspace in self.vector_stores:
            debug(self.config, f"Vector store for workspace '{workspace}' already loaded")
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
                dirs = get_workspace_dirs(workspace)
                vector_dir = Path(dirs['vector_store'])
                metadata_path = vector_dir / "metadata.json"
                success = metadata_path.exists()

            if success:
                # Store in cache
                self.vector_stores[workspace] = {
                    'loaded': datetime.datetime.now().isoformat(),
                }
                print(f"Vector store loaded for workspace '{workspace}'")

            return success
        except Exception as e:
            debug(self.config, f"Error loading vector store: {str(e)}")
            return False

    def get_workspace_stats(self, workspace):
        """
        Get statistics for a workspace

        Args:
            workspace (str): Target workspace

        Returns:
            dict: Workspace statistics
        """
        debug(self.config, f"Getting statistics for workspace: {workspace}")

        # Get directories using common.py function
        dirs = get_workspace_dirs(workspace)
        workspace_dir = Path(dirs['documents'])

        if not workspace_dir.exists():
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
            vector_dir = Path(dirs['vector_store'])
            metadata_path = vector_dir / "metadata.json"

            # Use load_json from utils.py
            metadata = load_json(metadata_path, {
                'created': datetime.datetime.now().isoformat(),
                'document_count': doc_count,
                'embedding_model': self.config.get('embedding.default_model'),
            })

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
            debug(self.config, f"Error getting workspace stats: {str(e)}")
            return None

    def delete_workspace(self, workspace):
        """
        Delete a workspace and all its data

        Args:
            workspace (str): Workspace to delete

        Returns:
            bool: Success flag
        """
        debug(self.config, f"Deleting workspace: {workspace}")

        try:
            # Get directories using common.py function
            dirs = get_workspace_dirs(workspace)

            # Delete data directory
            data_dir = Path(dirs['data'])
            if data_dir.exists():
                import shutil
                shutil.rmtree(data_dir)

            # Delete documents directory
            body_dir = Path(dirs['body'])
            if body_dir.exists():
                import shutil
                shutil.rmtree(body_dir)

            # Remove from cache
            if workspace in self.vector_stores:
                del self.vector_stores[workspace]

            print(f"Workspace '{workspace}' deleted")
            return True
        except Exception as e:
            print(f"Error deleting workspace: {str(e)}")
            return False

# Helper function
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
        debug(config, "Vector store inspector initialized")

    def inspect_workspace(self, workspace):
        """
        Inspect a workspace's vector store and data

        Args:
            workspace (str): Workspace to inspect
        """
        debug(self.config, f"Inspecting workspace: {workspace}")

        # Get directories using common.py function
        dirs = get_workspace_dirs(workspace)
        data_dir = Path(dirs['data'])
        body_dir = Path(dirs['body'])
        documents_dir = Path(dirs['documents'])
        vector_dir = Path(dirs['vector_store'])

        print(f"\nInspecting workspace: {workspace}")

        # Check workspace directories
        print("\nWorkspace Directories:")
        print(f"  Data Directory: {data_dir} - {'Exists' if data_dir.exists() else 'Missing'}")
        print(f"  Body Directory: {body_dir} - {'Exists' if body_dir.exists() else 'Missing'}")

        # Check document files
        if documents_dir.exists():
            document_files = [f.name for f in documents_dir.glob('*.json')]
            print(f"\nDocument JSON Files: {len(document_files)}")
            for i, file in enumerate(document_files[:5]):  # Show first 5
                print(f"  {i + 1}. {file}")

                # Peek inside the document file
                try:
                    doc = load_json(documents_dir / file)
                    if doc:
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

        print(f"\nVector Store Type: {vector_store_type}")
        print(f"Vector Store Directory: {vector_dir} - {'Exists' if vector_dir.exists() else 'Missing'}")

        # If vector store directory exists, list and check all files
        if vector_dir.exists():
            print("\nVector Store Files:")
            for file_path in vector_dir.iterdir():
                size = file_path.stat().st_size if file_path.is_file() else 0
                print(f"  {file_path.name}: {format_size(size)}")

                # For JSON files, validate and show basic structure
                if file_path.name.endswith('.json') and file_path.is_file():
                    try:
                        data = load_json(file_path)
                        if isinstance(data, dict):
                            print(f"    Keys: {list(data.keys())[:5]}{' ...' if len(data.keys()) > 5 else ''}")

                            # Count documents for docstore
                            if file_path.name == 'docstore.json' and 'docstore/docs' in data:
                                print(f"    Documents in docstore: {len(data['docstore/docs'])}")

                                # Show a sample document
                                if data['docstore/docs']:
                                    doc_id = next(iter(data['docstore/docs']))
                                    doc = data['docstore/docs'][doc_id]
                                    print(f"    Sample document ID: {doc_id}")
                                    if 'metadata' in doc:
                                        print(f"    Sample metadata: {doc.get('metadata', {}).get('source', 'unknown')}")
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

    # Other methods remain the same
