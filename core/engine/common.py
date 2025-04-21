"""
Common operations for document analysis toolkit
Standardized functions for frequently used operations
"""

import os
import datetime
import json
import shutil
from typing import Dict, Optional, Callable

from core.engine.logging import error, warning, info
from core.engine.dependencies import require
from core.engine.utils import ensure_dir, save_json, load_json


# ===== STANDARD PATHS AND LOCATIONS =====

def get_workspace_dirs(workspace: str) -> Dict[str, str]:
    """
    Get standard directories for a workspace

    Args:
        workspace: Name of the workspace

    Returns:
        Dict with standard directory paths
    """
    return {
        'data': str(Path("data") / workspace),
        'body': str(Path("body") / workspace),
        'logs': str(Path("logs") / workspace),
        'vector_store': str(Path("data") / workspace / "vector_store"),
        'documents': str(Path("data") / workspace / "documents"),
        'cache': str(Path("data") / workspace / "cache")
    }


def ensure_workspace_dirs(workspace: str) -> Dict[str, bool]:
    """
    Ensure workspace directories exist

    Args:
        workspace: Name of the workspace

    Returns:
        Dict with creation status for each directory
    """
    dirs = get_workspace_dirs(workspace)

    results = {}
    for name, path in dirs.items():
        results[name] = ensure_dir(path)

    return results


def get_latest_file(directory: str, pattern: str = None, extension: str = None) -> Optional[str]:
    """
    Get the most recent file in a directory, optionally filtered by pattern or extension

    Args:
        directory: Directory to search
        pattern: Optional filename pattern to match
        extension: Optional file extension to match

    Returns:
        Path to the latest file or None if not found
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        return None

    matching_files = []

    for file_path in dir_path.iterdir():
        # Skip directories
        if not file_path.is_file():
            continue

        # Check pattern match if specified
        if pattern and pattern not in file_path.name:
            continue

        # Check extension if specified
        if extension and not file_path.name.endswith(extension):
            continue

        matching_files.append((str(file_path), file_path.stat().st_mtime))

    if not matching_files:
        return None

    # Return the most recent file
    matching_files.sort(key=lambda x: x[1], reverse=True)
    return matching_files[0][0]


# ===== ERROR HANDLING =====

def safe_execute(func: Callable, *args, default_return=None, error_message=None, **kwargs):
    """
    Safely execute a function with error handling

    Args:
        func: Function to execute
        *args: Arguments for the function
        default_return: Value to return on failure
        error_message: Custom error message prefix
        **kwargs: Keyword arguments for the function

    Returns:
        Function result or default_return on failure
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        msg = error_message or f"Error executing {func.__name__}"
        error(f"{msg}: {str(e)}")
        return default_return


# ===== MODEL LOADING =====

def get_embedding_model(config, model_name=None):
    """
    Get an embedding model with consistent error handling

    Args:
        config: Configuration object
        model_name: Optional model name override

    Returns:
        Embedding model or None if unavailable
    """
    # Check dependencies
    if not require('transformers', 'embedding model') or not require('torch', 'embedding model'):
        warning("Cannot load embedding model: required dependencies not available")
        return None

    # Use provided model name or get from config
    if not model_name:
        model_name = config.get('embedding.default_model', 'multilingual-e5')

    # Map model name to Hugging Face model path
    model_path_map = {
        "multilingual-e5": "intfloat/multilingual-e5-large",
        "mixbread": "mixedbread-ai/mxbai-embed-large-v1",
        "bge": "BAAI/bge-m3",
        "jina-zh": "jinaai/jina-embeddings-v2-base-zh"
    }

    model_path = model_path_map.get(model_name, model_path_map["multilingual-e5"])

    # Special handling for Chinese models
    if model_name == "jina-zh" and not require('jieba', 'Chinese embedding model'):
        warning("jieba not available for Chinese embedding - using multilingual model instead")
        model_path = model_path_map["multilingual-e5"]

    try:
        from transformers import AutoModel, AutoTokenizer

        # Log model loading
        info(f"Loading embedding model: {model_name} ({model_path})")

        # Load model and tokenizer
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Determine hardware device
        device = get_optimal_device(config)
        if device != "cpu":
            info(f"Moving embedding model to {device}")
            model = model.to(device)

        return {
            "model": model,
            "tokenizer": tokenizer,
            "name": model_name,
            "device": device
        }
    except Exception as e:
        error(f"Error loading embedding model {model_name}: {str(e)}")
        return None


def get_optimal_device(config):
    """
    Determine the optimal device for model execution

    Args:
        config: Configuration object

    Returns:
        Device string ('cuda', 'mps', or 'cpu')
    """
    # Check if a specific device is set in config
    device_setting = config.get('hardware.device', 'auto')
    if device_setting != 'auto':
        return device_setting

    # Auto-detect the best device
    if not require('torch', 'hardware detection'):
        return 'cpu'

    try:
        import torch

        # Check for CUDA
        if torch.cuda.is_available():
            return 'cuda'

        # Check for MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
    except Exception as e:
        warning(f"Error detecting hardware capabilities: {str(e)}")

    # Default to CPU
    return 'cpu'


# ===== FILE OPERATIONS =====

def create_timestamped_backup(filepath: str) -> Optional[str]:
    """
    Create a timestamped backup of a file or directory

    Args:
        filepath: Path to the file or directory

    Returns:
        Path to backup or None if failed
    """
    path = Path(filepath)
    if not path.exists():
        warning(f"Cannot create backup - path does not exist: {filepath}")
        return None

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}_backup_{timestamp}"
    backup_path_obj = Path(backup_path)

    try:
        if path.is_dir():
            shutil.copytree(path, backup_path_obj)
        else:
            shutil.copy2(path, backup_path_obj)

        info(f"Created backup at: {backup_path}")
        return backup_path
    except Exception as e:
        error(f"Error creating backup of {filepath}: {str(e)}")
        return None


def save_analysis_results(workspace: str, analysis_type: str, results: Dict, timestamp=None):
    """
    Save analysis results with standard formatting

    Args:
        workspace: Name of the workspace
        analysis_type: Type of analysis (themes, topics, sentiment, etc.)
        results: Analysis results
        timestamp: Optional timestamp override

    Returns:
        Path to saved file
    """
    # Create logs directory
    logs_dir = Path("logs") / workspace
    ensure_dir(logs_dir)

    # Generate timestamp if not provided
    if not timestamp:
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Determine output format from configuration
    import core.engine.config as config_module
    if hasattr(config_module, 'Config'):
        config = config_module.Config()
        output_format = config.get('system.output_format', 'txt')
    else:
        output_format = 'txt'

    # Create metadata structure
    metadata = {
        'timestamp': datetime.datetime.now().isoformat(),
        'workspace': workspace,
        'analysis_type': analysis_type
    }

    # Combine with results
    output_data = {**metadata, 'results': results}

    # Generate filename
    filename = f"{analysis_type}_{timestamp}.{output_format}"
    filepath = logs_dir / filename

    # Save based on format
    if output_format == 'json':
        save_json(filepath, output_data)
    else:
        # Default to text format
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"=== {analysis_type.upper()} ANALYSIS ===\n\n")
            f.write(f"Timestamp: {metadata['timestamp']}\n")
            f.write(f"Workspace: {workspace}\n\n")

            # Write results in a readable format
            f.write(format_results_as_text(results))

    info(f"Saved {analysis_type} analysis results to: {filepath}")
    return filepath


def format_results_as_text(results: Dict) -> str:
    """
    Format results dictionary as readable text

    Args:
        results: Results dictionary

    Returns:
        Formatted text
    """
    lines = []

    def _format_dict(d, indent=0):
        indent_str = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{indent_str}{key}:")
                _format_dict(value, indent + 1)
            elif isinstance(value, list):
                lines.append(f"{indent_str}{key}:")
                _format_list(value, indent + 1)
            else:
                lines.append(f"{indent_str}{key}: {value}")

    def _format_list(lst, indent=0):
        indent_str = "  " * indent
        for i, item in enumerate(lst):
            if isinstance(item, dict):
                lines.append(f"{indent_str}Item {i + 1}:")
                _format_dict(item, indent + 1)
            elif isinstance(item, list):
                lines.append(f"{indent_str}Item {i + 1}:")
                _format_list(item, indent + 1)
            else:
                lines.append(f"{indent_str}- {item}")

    if isinstance(results, dict):
        _format_dict(results)
    elif isinstance(results, list):
        _format_list(results)
    else:
        lines.append(str(results))

    return "\n".join(lines)


# ===== WORKSPACE OPERATIONS =====

def check_workspace_exists(workspace: str) -> bool:
    """
    Check if a workspace exists

    Args:
        workspace: Name of the workspace

    Returns:
        True if workspace exists, False otherwise
    """
    dirs = get_workspace_dirs(workspace)
    return Path(dirs['data']).exists() or Path(dirs['body']).exists()


def create_workspace(workspace: str, confirm: bool = True) -> bool:
    """
    Create a new workspace with confirmation

    Args:
        workspace: Name of the workspace
        confirm: Whether to ask for confirmation

    Returns:
        True if workspace was created, False otherwise
    """
    # Check if workspace already exists
    if check_workspace_exists(workspace):
        info(f"Workspace '{workspace}' already exists")
        return True

    # Ask for confirmation if required
    if confirm:
        answer = input(f"Workspace '{workspace}' does not exist. Create it? (y/n): ").strip().lower()
        if answer not in ('y', 'yes'):
            info("Workspace creation cancelled")
            return False

    # Create workspace directories
    ensure_workspace_dirs(workspace)
    info(f"Created workspace: {workspace}")
    return True


def get_workspace_stats(workspace: str) -> Dict:
    """
    Get statistics for a workspace

    Args:
        workspace: Name of the workspace

    Returns:
        Dictionary with workspace statistics
    """
    dirs = get_workspace_dirs(workspace)
    stats = {
        'name': workspace,
        'exists': check_workspace_exists(workspace),
        'file_count': 0,
        'document_count': 0,
        'vector_store_exists': False,
        'last_updated': None,
        'languages': {}
    }

    # Count files in body directory
    if os.path.exists(dirs['body']):
        for root, _, files in os.walk(dirs['body']):
            stats['file_count'] += len(files)

    # Count documents in documents directory
    if os.path.exists(dirs['documents']):
        doc_files = [f for f in os.listdir(dirs['documents']) if f.endswith('.json')]
        stats['document_count'] = len(doc_files)

    # Check vector store
    stats['vector_store_exists'] = os.path.exists(dirs['vector_store'])

    # Get vector store metadata if available
    metadata_path = os.path.join(dirs['vector_store'], 'metadata.json')
    if os.path.exists(metadata_path):
        try:
            metadata = load_json(metadata_path, {})
            stats['last_updated'] = metadata.get('created', None)
            stats['embedding_model'] = metadata.get('embedding_model', None)
        except Exception as e:
            warning(f"Error reading vector store metadata: {str(e)}")

    # Analyze languages if documents exist
    if stats['document_count'] > 0:
        language_counts = {}

        # Sample up to 100 documents to determine language distribution
        doc_files = [f for f in os.listdir(dirs['documents']) if f.endswith('.json')]
        sample_count = min(100, len(doc_files))

        for filename in doc_files[:sample_count]:
            filepath = os.path.join(dirs['documents'], filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    doc = json.load(f)
                    language = doc.get('metadata', {}).get('language', 'unknown')
                    language_counts[language] = language_counts.get(language, 0) + 1
            except Exception:
                pass

        # Calculate percentages
        stats['languages'] = {
            lang: {
                'count': count,
                'percentage': (count / sample_count) * 100
            }
            for lang, count in language_counts.items()
        }

    return stats
