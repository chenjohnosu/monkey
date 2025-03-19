"""
File processing module for scanning directories and creating vector databases
"""

import os
import time
import hashlib
from datetime import datetime
from core.engine.utils import debug_print, is_supported_file, get_file_content, ensure_dir
from core.engine.storage import StorageManager
from core.language.processor import TextProcessor

class FileProcessor:
    """Processes files for document analysis"""

    def __init__(self, config, storage_manager=None, text_processor=None):
        """Initialize the file processor"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.text_processor = text_processor or TextProcessor(config)
        debug_print(config, "File processor initialized")

    def process_workspace(self, workspace, scan_only=False, force=False):
        """
        Process all files in a workspace

        Args:
            workspace (str): The workspace to process
            scan_only (bool): If True, only scan for new files without processing them
            force (bool): If True, process all files regardless of their change status
        """
        debug_print(self.config, f"Processing workspace: {workspace}" + (" (force mode)" if force else ""))

        # Ensure workspace directories exist
        data_dir = os.path.join("data", workspace)
        body_dir = os.path.join("body", workspace)

        if not os.path.exists(body_dir):
            print(f"Document directory does not exist: {body_dir}")
            return

        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)

        # If force is True, get all files in the workspace
        if force:
            files_to_process = self._get_all_files(workspace)
            print(f"Force mode: Processing all {len(files_to_process)} files in workspace '{workspace}'")
        else:
            # Scan for new or updated files
            files_to_process = self.scan_workspace(workspace, detailed=True, return_files=True)

            if not files_to_process:
                print(f"No new or updated files found in workspace '{workspace}'")
                return

        if scan_only:
            # If scan_only, we've already displayed the results in scan_workspace
            return

        print(f"Processing {len(files_to_process)} files in workspace '{workspace}'")

        # Initialize counters
        new_count = 0
        updated_count = 0
        skipped_count = 0
        failed_count = 0

        # Process each file
        for file_info in files_to_process:
            rel_path = file_info['rel_path']
            filepath = file_info['abs_path']
            status = file_info.get('status', "Force" if force else "Unknown")

            try:
                # Process the file
                self._process_file(workspace, rel_path, filepath)

                if status == "New":
                    new_count += 1
                    print(f"Added: {rel_path}")
                elif status == "Force":
                    updated_count += 1
                    print(f"Forced update: {rel_path}")
                else:
                    updated_count += 1
                    print(f"Updated: {rel_path}")

            except Exception as e:
                print(f"Failed to process {rel_path}: {str(e)}")
                failed_count += 1

        # Print summary
        print(f"\nProcessing complete:")
        print(f"  New files: {new_count}")
        print(f"  Updated files: {updated_count}")
        print(f"  Skipped files: {skipped_count}")
        print(f"  Failed files: {failed_count}")

        # Create or update vector store
        if new_count > 0 or updated_count > 0:
            self.storage_manager.create_vector_store(workspace)

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
                    modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
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

                modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
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

        # Create document metadata
        metadata = {
            'source': rel_path,
            'workspace': workspace,
            'language': processed['language'],
            'tokens': processed['tokens'],
            'content_hash': content_hash,
            'last_modified': os.path.getmtime(filepath),
            'file_size': os.path.getsize(filepath),
            'processed_date': datetime.now().isoformat()
        }

        # Add to storage
        self.storage_manager.add_document(workspace, rel_path, content, processed['processed'], metadata)

        # Return document info
        return {
            'path': rel_path,
            'language': processed['language'],
            'tokens': processed['tokens'],
            'hash': content_hash
        }