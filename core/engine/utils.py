"""
Streamlined utility functions for the document analysis toolkit
"""

import os
import datetime
import re
import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional

from core.engine.logging import debug, error, warning, info

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path.exists()

def get_file_extension(filepath):
    """Get the extension of a file"""
    return Path(filepath).suffix.lower()

def get_supported_extensions():
    """Get a list of supported file extensions"""
    return ['.txt', '.md', '.pdf', '.docx', '.html']

def is_supported_file(filepath):
    """Check if a file is supported"""
    return get_file_extension(filepath) in get_supported_extensions()

def get_file_content(filepath):
    """Get the content of a file as text"""
    if not Path(filepath).exists():
        error(f"File does not exist: {filepath}")
        return None

    ext = get_file_extension(filepath)

    # Text files (TXT, MD)
    if ext in ['.txt', '.md']:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                error(f"Failed to read file with latin-1 encoding: {str(e)}")
                return None

    # DOCX files
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(filepath)
            return '\n'.join(para.text for para in doc.paragraphs)
        except ImportError:
            error("python-docx library not installed.")
            return None
        except Exception as e:
            error(f"Failed to extract DOCX content: {str(e)}")
            return None

    # PDF files
    elif ext == '.pdf':
        try:
            import PyPDF2
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return '\n'.join(page.extract_text() for page in reader.pages)
        except ImportError:
            try:
                from pdfminer.high_level import extract_text
                return extract_text(filepath)
            except ImportError:
                error("PDF extraction libraries not installed.")
                return None
        except Exception as e:
            error(f"Failed to extract PDF content: {str(e)}")
            return None

    # HTML files
    elif ext == '.html':
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                html_content = file.read()
            return re.sub('<[^<]+?>', ' ', html_content).strip()
        except Exception as e:
            error(f"Failed to extract HTML content: {str(e)}")
            return None
    else:
        return None

def save_json(filepath, data, indent=2):
    """Save data to a JSON file"""
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)
        return True
    except Exception as e:
        error(f"Error saving JSON file: {str(e)}")
        return False

def load_json(filepath, default=None):
    """Load data from a JSON file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        warning(f"JSON file not found: {filepath}")
        return default
    except json.JSONDecodeError:
        error(f"Error parsing JSON file: {filepath}")
        return default
    except Exception as e:
        error(f"Error loading JSON file: {str(e)}")
        return default

def create_timestamped_backup(filepath):
    """Create a timestamped backup of a file or directory"""
    path = Path(filepath)
    if not path.exists():
        return None

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}_backup_{timestamp}"

    try:
        if path.is_dir():
            shutil.copytree(path, backup_path)
        else:
            shutil.copy2(path, backup_path)
        return backup_path
    except Exception as e:
        error(f"Error creating backup: {str(e)}")
        return None

def timestamp_filename(prefix, extension):
    """Generate a filename with a timestamp"""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{prefix}_{timestamp}.{extension}"

def format_size(size_bytes):
    """Format file size in a human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024 or unit == 'GB':
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024

def calculate_file_hash(filepath):
    """Calculate MD5 hash of a file's content"""
    content = get_file_content(filepath)
    if content:
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    return hashlib.md5(str(os.path.getmtime(filepath)).encode('utf-8')).hexdigest()

def split_text_into_chunks(text, chunk_size=500):
    """Split text into chunks of approximately equal size"""
    if not text:
        return []

    # Use sentence splitting for more natural chunks
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

def configure_vectorizer(config, doc_count, language=None, stopwords=None):
    """Configure a TF-IDF vectorizer for the given corpus"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        error("scikit-learn not available for vectorization")
        return None

    # Determine min/max document frequency based on corpus size
    if doc_count <= 5:
        min_df, max_df = 1, 1.0
    elif doc_count <= 20:
        min_df, max_df = 1, 0.8
    else:
        min_df, max_df = 2, 0.7

    # Create vectorizer with appropriate parameters
    if language == 'zh':
        from core.language.tokenizer import ChineseTokenizer
        tokenizer = ChineseTokenizer(stopwords)
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            tokenizer=tokenizer
        )
    else:
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words="english"
        )

    return vectorizer

def extract_keywords(config, texts, language="en", top_n=10, stopwords=None):
    """Extract keywords from a collection of texts using TF-IDF"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
    except ImportError:
        error("scikit-learn not available for keyword extraction")
        return []

    if not texts:
        return []

    # Get vectorizer
    vectorizer = configure_vectorizer(config, len(texts), language, stopwords)
    if not vectorizer:
        return []

    try:
        # Create document-term matrix
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score for each term
        avg_scores = np.asarray(X.mean(axis=0)).ravel()

        # Sort terms by score
        sorted_indices = avg_scores.argsort()[::-1]

        # Extract keywords, filtering stopwords
        keywords = []
        for idx in sorted_indices:
            term = feature_names[idx]
            if stopwords and term in stopwords:
                continue
            keywords.append(term)
            if len(keywords) >= top_n:
                break

        return keywords
    except Exception as e:
        error(f"Error extracting keywords: {str(e)}")
        return []

# Compatibility functions to support older modules
def format_feedback(message: str, success: bool = True) -> str:
    """Format feedback message for compatibility with older modules"""
    status = "✓" if success else "✗"
    return f"{status} {message}"

def format_header(title: str) -> str:
    """Format a section header for compatibility with older modules"""
    return f"\n{title}\n{'=' * len(title)}"

def format_subheader(title: str) -> str:
    """Format a subsection header for compatibility with older modules"""
    return f"\n{title}\n{'-' * len(title)}"

def format_mini_header(title: str) -> str:
    """Format a mini header for compatibility with older modules"""
    return f"\n{title}:"

def format_key_value(key: str, value: Any, indent: int = 0) -> str:
    """Format a key-value pair for compatibility with older modules"""
    spaces = " " * indent
    return f"{spaces}{key}: {value}"

def format_list_item(text: str, indent: int = 0) -> str:
    """Format a list item for compatibility with older modules"""
    spaces = " " * indent
    return f"{spaces}• {text}"

def format_code_block(content: str, indent: int = 0) -> str:
    """Format a code block for compatibility with older modules"""
    spaces = " " * indent
    lines = content.split('\n')
    formatted_lines = [f"{spaces}{line}" for line in lines]
    return '\n'.join(formatted_lines)