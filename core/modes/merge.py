"""
Vector store merging module
"""

import os
from core.engine.logging import debug
from core.engine.storage import StorageManager

class VectorStoreMerger:
    """Merges vector stores from different workspaces"""
    
    def __init__(self, config, storage_manager=None):
        """Initialize the vector store merger"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        debug(config, "Vector store merger initialized")

    def merge(self, source_workspace, dest_workspace):
        """
        Merge source workspace into destination workspace

        Args:
            source_workspace (str): Source workspace name
            dest_workspace (str): Destination workspace name
        """
        debug(self.config, f"Merging workspace '{source_workspace}' into '{dest_workspace}'")

        # Check if source workspace exists
        source_data_dir = os.path.join("data", source_workspace)
        if not os.path.exists(source_data_dir):
            print(f"Source workspace '{source_workspace}' does not exist or has no vector store")
            return

        # Check if destination workspace exists, create if not
        dest_data_dir = os.path.join("data", dest_workspace)
        dest_body_dir = os.path.join("body", dest_workspace)

        for directory in [dest_data_dir, dest_body_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

        # Get documents from source workspace
        source_docs = self.storage_manager.get_documents(source_workspace)
        if not source_docs:
            print(f"No documents found in source workspace '{source_workspace}'")
            return

        print(f"Found {len(source_docs)} documents in source workspace '{source_workspace}'")

        # Get existing documents in destination workspace
        dest_docs = self.storage_manager.get_documents(dest_workspace)
        existing_count = len(dest_docs) if dest_docs else 0

        print(f"Found {existing_count} existing documents in destination workspace '{dest_workspace}'")

        # Prepare for merging
        merged_count = 0
        duplicate_count = 0

        # Track source files by path
        dest_paths = {doc['metadata']['source'] for doc in dest_docs} if dest_docs else set()

        # Add documents from source to destination
        for doc in source_docs:
            source_path = doc['metadata']['source']

            # Check for duplicates by path
            if source_path in dest_paths:
                duplicate_count += 1
                debug(self.config, f"Skipping duplicate document: {source_path}")
                continue

            # Update metadata
            doc['metadata']['workspace'] = dest_workspace
            doc['metadata']['merged_from'] = source_workspace

            # Add to destination
            self.storage_manager.add_document(
                dest_workspace,
                source_path,
                doc['content'],
                doc['processed_content'],
                doc['metadata']
            )

            merged_count += 1
            dest_paths.add(source_path)

        # Create or update vector store for destination workspace
        if merged_count > 0:
            self.storage_manager.create_vector_store(dest_workspace)

        # Print summary
        print(f"\nMerge complete:")
        print(f"  Documents merged: {merged_count}")
        print(f"  Duplicates skipped: {duplicate_count}")
        print(f"  Total documents in destination: {existing_count + merged_count}")