"""
Consolidated utility functions for the document analysis toolkit
"""

import os
import time
import datetime
import re
import json
from typing import Dict, List, Any
from pathlib import Path
import tempfile
import shutil
import contextlib
from functools import wraps

# Import logging for debug_print usage
from core.engine.logging import debug_print, error, warning, info, debug, trace


# ===== FILE OPERATIONS =====

def ensure_dir(directory):
    """Ensure a directory exists, creating it if necessary using pathlib"""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return not path.exists()

def get_file_extension(filepath):
    """Get the extension of a file using pathlib"""
    return Path(filepath).suffix.lower()

def get_supported_extensions():
    """Get a list of supported file extensions"""
    return ['.txt', '.md', '.pdf', '.docx', '.html']

def is_supported_file(filepath):
    """Check if a file is supported"""
    return get_file_extension(filepath) in get_supported_extensions()

def get_file_content(filepath):
    """Get the content of a file as text"""
    ext = get_file_extension(filepath)

    # Text files (TXT, MD)
    if ext in ['.txt', '.md']:
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                return file.read()
        except UnicodeDecodeError:
            # Try with different encodings
            try:
                with open(filepath, 'r', encoding='latin-1') as file:
                    return file.read()
            except Exception as e:
                debug_print(None, f"Failed to read file with latin-1 encoding: {filepath}, {str(e)}")
                return None

    # DOCX files
    elif ext == '.docx':
        try:
            import docx
            doc = docx.Document(filepath)
            full_text = []
            # Extract text from paragraphs
            for para in doc.paragraphs:
                full_text.append(para.text)
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        full_text.append(cell.text)
            return '\n'.join(full_text)
        except ImportError:
            debug_print(None, f"python-docx library not installed. Install with: pip install python-docx")
            return f"ERROR: python-docx not installed. Cannot extract DOCX content from {filepath}"
        except Exception as e:
            debug_print(None, f"Failed to extract DOCX content: {filepath}, {str(e)}")
            return None

    # PDF files
    elif ext == '.pdf':
        try:
            import PyPDF2
            pdf_text = []
            with open(filepath, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    pdf_text.append(page.extract_text())
            return '\n'.join(pdf_text)
        except ImportError:
            try:
                # Alternative using pdfminer if PyPDF2 is not available
                from pdfminer.high_level import extract_text
                return extract_text(filepath)
            except ImportError:
                debug_print(None,
                            f"PDF extraction libraries not installed. Install with: pip install PyPDF2 or pip install pdfminer.six")
                return f"ERROR: PDF extraction libraries not installed. Cannot extract PDF content from {filepath}"
        except Exception as e:
            debug_print(None, f"Failed to extract PDF content: {filepath}, {str(e)}")
            return None

    # HTML files
    elif ext == '.html':
        try:
            # Extract text from HTML
            with open(filepath, 'r', encoding='utf-8') as file:
                html_content = file.read()

            # Simple HTML tag removal
            text = re.sub('<[^<]+?>', ' ', html_content)
            text = re.sub('\\s+', ' ', text)
            return text.strip()
        except Exception as e:
            debug_print(None, f"Failed to extract HTML content: {filepath}, {str(e)}")
            return None
    else:
        return None


def save_json(filepath, data, indent=2):
    """Save data to a JSON file with proper error handling and atomic writes"""
    try:
        # Create parent directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        # Write to temp file first, then rename for atomicity
        temp_file = f"{filepath}.tmp"
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent)

        # Atomic replace
        os.replace(temp_file, filepath)
        return True
    except Exception as e:
        error(f"Error saving JSON file {filepath}: {str(e)}")
        # Clean up temp file if it exists
        with contextlib.suppress(FileNotFoundError):
            os.remove(f"{filepath}.tmp")
        return False


def load_json(filepath, default=None):
    """Load data from a JSON file with error handling"""
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
        error(f"Error loading JSON file {filepath}: {str(e)}")
        return default


def create_timestamped_backup(filepath):
    """Create a timestamped backup of a file or directory using proper temp files"""
    path = Path(filepath)
    if not path.exists():
        return None

    # Create backup filename with timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_path = f"{filepath}_backup_{timestamp}"

    try:
        # Use temporary directory for atomic operation
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_backup = Path(tmpdir) / path.name

            # Copy to temp location first
            if path.is_dir():
                shutil.copytree(path, temp_backup)
            else:
                shutil.copy2(path, temp_backup)

            # Then move to final location (more atomic)
            shutil.move(temp_backup, backup_path)

        return backup_path
    except Exception as e:
        error(f"Error creating backup of {filepath}: {str(e)}")
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

def load_original_document(config, workspace, source_path):
    """
    Load original document content from body directory with fallback handling

    Args:
        config: Configuration object for debug output
        workspace (str): Workspace name
        source_path (str): Document source path within workspace

    Returns:
        str: Original document content or None if not available
    """
    debug_print(config, f"Loading original document: {source_path}")

    try:
        # Construct path to original file
        original_path = os.path.join("body", workspace, source_path)

        if os.path.exists(original_path):
            # Get content based on file type
            content = get_file_content(original_path)
            if content:
                debug_print(config, f"Successfully loaded original content ({len(content)} chars)")
                return content
            else:
                debug_print(config, f"Failed to extract content from {original_path}")
        else:
            debug_print(config, f"Original file not found: {original_path}")
    except Exception as e:
        debug_print(config, f"Error loading original document: {str(e)}")

    return None


# ===== TEXT PROCESSING =====

def split_text_into_chunks(text, chunk_size=500):
    """
    Split text into chunks more efficiently

    Args:
        text (str): Text to split
        chunk_size (int): Maximum chunk size

    Returns:
        List[str]: Text chunks
    """
    if not text:
        return []

    # Use regular expressions for more efficient sentence splitting
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence would exceed chunk size
        if len(current_chunk) + len(sentence) + 1 > chunk_size:
            # Save current chunk if it exists
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Handle sentences longer than chunk_size
            if len(sentence) > chunk_size:
                # Split long sentence into chunks
                words = sentence.split()
                current_chunk = ""

                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += " " + word
                    else:
                        chunks.append(current_chunk.strip())
                        current_chunk = word
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            current_chunk += " " + sentence

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def configure_vectorizer(config, doc_count, language=None, chinese_stopwords=None):
    """
    Create a TF-IDF vectorizer with parameters appropriate for corpus size and language

    Args:
        config: Configuration object for debug output
        doc_count (int): Number of documents in corpus
        language (str): Primary language of corpus
        chinese_stopwords (set): Set of Chinese stopwords

    Returns:
        TfidfVectorizer: Configured vectorizer
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except ImportError:
        error("scikit-learn not available. Install with: pip install scikit-learn")
        return None

    debug_print(config, f"Configuring TF-IDF vectorizer for {doc_count} documents in language: {language}")

    # Determine if Chinese corpus
    is_chinese = language == "zh"

    # Configure min_df and max_df based on corpus size
    if doc_count <= 2:
        # For tiny corpus
        min_df = 1
        max_df = 1.0
    elif doc_count <= 5:
        # For very small corpus
        min_df = 1
        max_df = 0.9
    elif doc_count <= 10:
        # For small corpus
        min_df = 1
        max_df = 0.8
    elif doc_count <= 50:
        # For medium corpus
        min_df = 2
        max_df = 0.8
    else:
        # For large corpus
        min_df = 2
        max_df = 0.7

    debug_print(config, f"Using min_df={min_df}, max_df={max_df}")

    # Configure language-specific parameters
    if is_chinese:
        try:
            from core.language.tokenizer import ChineseTokenizer
            tokenizer = ChineseTokenizer()

            # Create vectorizer for Chinese text
            vectorizer = TfidfVectorizer(
                min_df=min_df,
                max_df=max_df,
                stop_words=None,
                # Remove token_pattern when using custom tokenizer
                tokenizer=tokenizer
            )
        except ImportError:
            error("ChineseTokenizer not available")
            return None
    else:
        # Create vectorizer for English text
        vectorizer = TfidfVectorizer(
            min_df=min_df,
            max_df=max_df,
            stop_words="english"
        )

    return vectorizer


def extract_keywords(config, texts, language="en", top_n=10, stopwords=None):
    """
    Extract keywords from a collection of texts using TF-IDF

    Args:
        config: Configuration object
        texts (List[str]): Collection of texts
        language (str): Language of texts
        top_n (int): Number of keywords to extract
        stopwords (Set[str]): Stopwords to filter out

    Returns:
        List[str]: Extracted keywords
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
    except ImportError:
        error("scikit-learn not available. Install with: pip install scikit-learn")
        return []

    debug_print(config, f"Extracting keywords from {len(texts)} texts in {language}")

    if not texts:
        return []

    # Get vectorizer configured for corpus
    vectorizer = configure_vectorizer(config, len(texts), language, stopwords)
    if not vectorizer:
        return []

    try:
        # Create document-term matrix
        X = vectorizer.fit_transform(texts)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # Calculate average TF-IDF score for each term
        avg_scores = np.asarray(X.mean(axis=0)).ravel()

        # Sort terms by score
        scored_terms = list(zip(feature_names, avg_scores))
        sorted_terms = sorted(scored_terms, key=lambda x: x[1], reverse=True)

        # Filter stopwords if provided
        if stopwords:
            sorted_terms = [(term, score) for term, score in sorted_terms
                           if term not in stopwords]

        # Extract keywords
        keywords = [term for term, score in sorted_terms[:top_n]]

        return keywords
    except Exception as e:
        debug_print(config, f"Error extracting keywords: {str(e)}")
        return []


# ===== PERFORMANCE MEASUREMENT =====

def measure_execution_time(func):
    """Decorator to measure execution time of a function using functools.wraps"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        print(f"Function '{func.__name__}' executed in {execution_time:.2f} seconds")
        return result
    return wrapper


# ===== FORMATTING UTILITIES =====

# ANSI color codes
class Colors:
    RESET = "\033[0m"
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    GRAY = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Styles
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"

    # Backgrounds
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"


def format_debug(message: str) -> str:
    """Format debug message with timestamp and gray color"""
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return f"{Colors.GRAY}[DEBUG {timestamp}] {message}{Colors.RESET}"

def format_header(title: str) -> str:
    """Format a section header with bold magenta"""
    return f"\n{Colors.BOLD}{Colors.MAGENTA}{title}{Colors.RESET}"


def format_subheader(title: str) -> str:
    """Format a subsection header with cyan"""
    return f"\n{Colors.CYAN}{title}{Colors.RESET}"


def format_mini_header(title: str) -> str:
    """Format a mini header with yellow"""
    return f"\n{Colors.YELLOW}{title}{Colors.RESET}"


def format_command(command: str) -> str:
    """Format a command with bright blue"""
    return f"{Colors.BRIGHT_BLUE}CMD> {command}{Colors.RESET}"


def format_feedback(message: str, success: bool = True) -> str:
    """Format system feedback message with green for success, red for errors"""
    color = Colors.GREEN if success else Colors.RED
    status = "✓" if success else "✗"
    return f"{color}{status} {message}{Colors.RESET}"


def format_key_value(key: str, value: Any, indent: int = 0) -> str:
    """Format a key-value pair with indentation"""
    spaces = " " * indent
    return f"{spaces}{Colors.BRIGHT_WHITE}{key}:{Colors.RESET} {value}"


def format_list_item(text: str, indent: int = 0) -> str:
    """Format a list item with indentation"""
    spaces = " " * indent
    return f"{spaces}• {text}"


def format_analysis_result(title: str, content: str) -> str:
    """Format analysis result with bright white title and normal content"""
    return f"{Colors.BRIGHT_WHITE}{title}:{Colors.RESET} {content}"


def format_code_block(content: str, indent: int = 0) -> str:
    """
    Format a code block with appropriate styling based on mode (TUI or console)

    Args:
        content (str): The text content to format
        indent (int): Indentation level

    Returns:
        str: Formatted text
    """
    # Check if we're running in TUI mode by checking a global flag from logging module
    from core.engine.logging import _in_tui_mode

    spaces = " " * indent
    lines = content.split('\n')

    if _in_tui_mode:
        # In TUI mode, don't use ANSI color codes directly
        # Use markup that the textual library can interpret properly
        formatted_lines = [f"{spaces}{line}" for line in lines]
    else:
        # In console mode, use ANSI color codes as before
        from core.engine.utils import Colors
        formatted_lines = [f"{spaces}{Colors.GRAY}{line}{Colors.RESET}" for line in lines]

    return '\n'.join(formatted_lines)


# ===== MARKDOWN FORMATTING =====

def format_md_header(title: str) -> str:
    """Format markdown header"""
    return f"\n## {title}"


def format_md_subheader(title: str) -> str:
    """Format markdown subheader"""
    return f"\n### {title}"


def format_md_mini_header(title: str) -> str:
    """Format markdown mini header"""
    return f"\n#### {title}"


def format_md_command(command: str) -> str:
    """Format markdown command"""
    return f"```bash\n{command}\n```"


def format_md_feedback(message: str, success: bool = True) -> str:
    """Format markdown system feedback"""
    status = "✅" if success else "❌"
    return f"**{status} {message}**"


def format_md_analysis(title: str, items: List[Dict[str, Any]]) -> str:
    """Format markdown analysis results"""
    md_lines = [f"### {title}\n"]

    for i, item in enumerate(items):
        if i > 0:
            md_lines.append("")  # Add empty line between items

        for k, v in item.items():
            md_lines.append(f"**{k}**: {v}")

    return "\n".join(md_lines)