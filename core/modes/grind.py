"""
File processing module for scanning directories and creating vector databases
"""

import os
import time
import hashlib
import json
from core.engine.utils import is_supported_file, get_file_content, ensure_dir
from core.engine.logging import debug_print
from core.engine.storage import StorageManager
from core.language.processor import TextProcessor
import datetime

class FileProcessor:
    """Processes files for document analysis"""

    def __init__(self, config, storage_manager=None, text_processor=None):
        """Initialize the file processor"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.text_processor = text_processor or TextProcessor(config)
        debug_print(config, "File processor initialized")

    def process_workspace(self, workspace):
        """
        Process all files in a workspace to create initial database

        Args:
            workspace (str): The workspace to process
        """
        debug_print(self.config, f"Processing workspace: {workspace}")

        # Ensure workspace directories exist
        data_dir = os.path.join("data", workspace)
        body_dir = os.path.join("body", workspace)

        if not os.path.exists(body_dir):
            print(f"Document directory does not exist: {body_dir}")
            return

        # Check if vector store already exists
        vector_store_dir = os.path.join(data_dir, "vector_store")
        if os.path.exists(vector_store_dir):
            print(f"Vector store already exists for workspace '{workspace}'.")
            print("To add new files or update existing ones, please use the /run update command.")
            return

        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # Get all files in the workspace
        files_to_process = self._get_all_files(workspace)

        # Check if there are files to process
        if not files_to_process:
            print(f"No files found in workspace '{workspace}'. Nothing to process.")
            return

        print(f"Creating initial database with {len(files_to_process)} source files in workspace '{workspace}'")

        # Process each file individually
        processed_files = []
        for file_info in files_to_process:
            rel_path = file_info['rel_path']
            filepath = file_info['abs_path']

            try:
                # Process the file
                processed_doc = self._process_file(workspace, rel_path, filepath)
                processed_files.append(processed_doc)
                print(f"Added file: {rel_path}")

            except Exception as e:
                print(f"Failed to process {rel_path}: {str(e)}")

        # Create vector store if any files were processed
        if processed_files:
            print(f"\nCreating vector store with {len(processed_files)} processed files...")
            success = self.storage_manager.create_vector_store(workspace)

            if success:
                print(f"Initial database created successfully for workspace '{workspace}'")
        else:
            print("No files were successfully processed. Vector store creation skipped.")

    def update_workspace(self, workspace):
        """
        Update workspace with new or modified files, and handle deleted files

        Args:
            workspace (str): The workspace to update
        """
        debug_print(self.config, f"Updating workspace: {workspace}")

        # Get list of all currently processed files in the database
        existing_files = self.storage_manager.get_processed_files(workspace)
        existing_paths = set(existing_files.keys())

        # Get list of all files currently in the file system
        body_dir = os.path.join("body", workspace)
        current_files = set()

        if os.path.exists(body_dir):
            for root, _, filenames in os.walk(body_dir):
                for filename in filenames:
                    filepath = os.path.join(root, filename)
                    if is_supported_file(filepath):
                        rel_path = os.path.relpath(filepath, body_dir)
                        current_files.add(rel_path)

        # Identify deleted files
        deleted_files = existing_paths - current_files

        # Scan for new or updated files
        files_to_update = self.scan_workspace(workspace, detailed=True, return_files=True)

        # Check if any file changes were detected
        if not files_to_update and not deleted_files:
            print(f"No file changes detected in workspace '{workspace}'")
            return

        # Process deletions first
        if deleted_files:
            print(f"Found {len(deleted_files)} deleted files in workspace '{workspace}'")
            for rel_path in deleted_files:
                try:
                    # Remove from document storage and vector store
                    self.storage_manager.remove_document(workspace, rel_path)
                    print(f"Removed: {rel_path}")
                except Exception as e:
                    print(f"Failed to remove {rel_path}: {str(e)}")

        # Process new/modified files
        if files_to_update:
            print(f"Processing {len(files_to_update)} new or modified files in workspace '{workspace}'")

            # Track successfully processed files
            processed_files = []

            # Process each file
            for file_info in files_to_update:
                rel_path = file_info['rel_path']
                filepath = file_info['abs_path']
                status = file_info.get('status', "Unknown")

                try:
                    # Remove existing entries for updated files
                    if status == "Modified":
                        self.storage_manager.remove_document(workspace, rel_path)

                    # Process the file
                    processed_doc = self._process_file(workspace, rel_path, filepath)
                    processed_files.append(processed_doc)

                    if status == "New":
                        print(f"Added: {rel_path}")
                    else:
                        print(f"Updated: {rel_path}")

                except Exception as e:
                    print(f"Failed to process {rel_path}: {str(e)}")

        # Update vector store if there were any changes
        if deleted_files or files_to_update:
            print(f"\nRebuilding vector store...")
            change_summary = []
            if deleted_files:
                change_summary.append(f"{len(deleted_files)} deletions")
            if files_to_update:
                change_summary.append(f"{len(files_to_update)} updates/additions")

            print(f"Vector store changes: {', '.join(change_summary)}")
            success = self.storage_manager.create_vector_store(workspace)

            if success:
                print(f"Workspace '{workspace}' updated successfully")
            else:
                print(f"Failed to update vector store for workspace '{workspace}'")
        else:
            print("No file changes were successfully processed. Vector store update skipped.")

    def scan_workspace(self, workspace, detailed=True, return_files=False):
        """
        Scan a workspace for new or modified files

        Args:
            workspace (str): The workspace to scan
            detailed (bool): Whether to print detailed results
            return_files (bool): Whether to return the list of files

        Returns:
            list: Files to process (if return_files=True) or None
        """
        debug_print(self.config, f"Scanning workspace: {workspace}")

        body_dir = os.path.join("body", workspace)

        if not os.path.exists(body_dir):
            print(f"Document directory does not exist: {body_dir}")
            return None if not return_files else []

        # Get processed files with metadata
        processed_files = self.storage_manager.get_processed_files(workspace)

        # Create mapping of file paths to content hashes
        hash_map = {}
        for path, metadata in processed_files.items():
            if 'content_hash' in metadata:
                hash_map[path] = metadata['content_hash']

        # Scan all files
        all_files = []
        for root, _, filenames in os.walk(body_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, body_dir)

                # Skip unsupported files
                if not is_supported_file(filepath):
                    continue

                # Check file status
                file_status = self._get_file_status(filepath, rel_path, hash_map)

                # Add to list if new or modified
                if file_status != "Unchanged":
                    modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                    size = os.path.getsize(filepath)

                    all_files.append({
                        'rel_path': rel_path,
                        'abs_path': filepath,
                        'status': file_status,
                        'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'size': self._format_size(size)
                    })

        # Sort by path
        all_files = sorted(all_files, key=lambda x: x['rel_path'])

        # Print detailed results if requested
        if detailed:
            if not all_files:
                print(f"No new or modified files found in workspace '{workspace}'")
            else:
                print(f"New or modified files in workspace '{workspace}':")
                for file_info in all_files:
                    print(f"  [{file_info['status']}] {file_info['rel_path']} ({file_info['size']}, {file_info['modified']})")

        return all_files if return_files else None

    def _get_all_files(self, workspace):
        """
        Get all files in a workspace, marking them all for processing

        Args:
            workspace (str): The workspace to scan

        Returns:
            list: All files in the workspace
        """
        debug_print(self.config, f"Getting all files in workspace: {workspace}")

        body_dir = os.path.join("body", workspace)
        if not os.path.exists(body_dir):
            return []

        all_files = []
        for root, _, filenames in os.walk(body_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, body_dir)

                # Skip unsupported files
                if not is_supported_file(filepath):
                    continue

                modified_time = datetime.datetime.fromtimestamp(os.path.getmtime(filepath))
                size = os.path.getsize(filepath)

                all_files.append({
                    'rel_path': rel_path,
                    'abs_path': filepath,
                    'status': 'Force',
                    'modified': modified_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'size': self._format_size(size)
                })

        return sorted(all_files, key=lambda x: x['rel_path'])

    def _get_file_status(self, filepath, rel_path, hash_map):
        """
        Determine if a file is new, modified, or unchanged

        Args:
            filepath (str): Path to the file
            rel_path (str): Relative path in the workspace
            hash_map (dict): Mapping of file paths to content hashes

        Returns:
            str: "New", "Modified", or "Unchanged"
        """
        # Check if file exists in processed files
        if rel_path not in hash_map:
            return "New"

        # Calculate current content hash
        current_hash = self._calculate_file_hash(filepath)

        # Compare with stored hash
        if current_hash != hash_map.get(rel_path):
            return "Modified"

        return "Unchanged"

    def _calculate_file_hash(self, filepath):
        """
        Calculate a hash of file content

        Args:
            filepath (str): Path to the file

        Returns:
            str: Content hash
        """
        try:
            content = get_file_content(filepath)
            if content:
                return hashlib.md5(content.encode('utf-8')).hexdigest()
        except Exception:
            pass

        # Fallback to modification time if content extraction fails
        return str(os.path.getmtime(filepath))

    def _format_size(self, size_bytes):
        """Format file size in a human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

    def _process_file(self, workspace, rel_path, filepath):
        """
        Process a single file

        Args:
            workspace (str): The workspace
            rel_path (str): Relative path within the workspace
            filepath (str): Full file path
        """
        debug_print(self.config, f"Processing file: {rel_path}")

        # Get file content
        content = get_file_content(filepath)
        if not content:
            raise ValueError("Failed to extract content")

        # Calculate content hash
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()

        # Preprocess text
        processed = self.text_processor.preprocess(content)

        # Select language-appropriate embedding model
        if processed['language'] == 'zh':
            embedding_model = self.config.get('embedding.chinese_model', 'jina-zh')
            print(f"🇨🇳 Chinese content detected. Using {embedding_model} embedding model for: {rel_path}")
        else:
            embedding_model = self.config.get('embedding.default_model', 'multilingual-e5')

        # Create document metadata
        metadata = {
            'source': rel_path,
            'workspace': workspace,
            'language': processed['language'],
            'embedding_model': embedding_model,  # Store which model was used
            'tokens': processed['tokens'],
            'content_hash': content_hash,
            'last_modified': os.path.getmtime(filepath),
            'file_size': os.path.getsize(filepath),
            'processed_date': datetime.datetime.now().isoformat()
        }

        # Add special indicator for Chinese documents using Jina model
        if processed['language'] == 'zh' and embedding_model == 'jina-zh':
            metadata['using_specialized_chinese_model'] = True
            debug_print(self.config, f"Using specialized jina-zh model for Chinese document: {rel_path}")

        # Add to storage
        self.storage_manager.add_document(workspace, rel_path, content, processed['processed'], metadata)

        # Return document info
        return {
            'path': rel_path,
            'language': processed['language'],
            'embedding_model': embedding_model,
            'tokens': processed['tokens'],
            'hash': content_hash
        }