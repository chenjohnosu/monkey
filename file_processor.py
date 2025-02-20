import os
import re
import logging
import PyPDF2
import docx2txt
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import spacy
from llama_index.core import Document
import hashlib

# Import MonkeyConfig
from config import MonkeyConfig


class TextPreprocessor:
    """Handles advanced text preprocessing including stop word removal."""

    def __init__(self):
        """Initialize the text preprocessor with necessary NLTK downloads and spaCy model."""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            self.nlp = spacy.load('en_core_web_sm')
            self.stop_words = self._get_enhanced_stopwords()
        except Exception as e:
            logging.error(f"Error initializing TextPreprocessor: {str(e)}")
            raise

    def _get_enhanced_stopwords(self) -> set[str]:
        """Create an enhanced set of stop words combining NLTK, spaCy, and custom words."""
        # Combine stop words from multiple sources
        stop_words = set(stopwords.words('english'))
        spacy_stops = set(self.nlp.Defaults.stop_words)

        # Custom academic/research stop words
        custom_stops = {
            'fig', 'figure', 'table', 'et', 'al', 'ie', 'eg', 'example',
            'paper', 'study', 'research', 'method', 'data', 'analysis',
            'result', 'discussion', 'conclusion', 'abstract', 'introduction',
            'background', 'methodology', 'appendix', 'copyright', 'journal',
            'vol', 'volume', 'issue', 'page', 'pp', 'doi'
        }

        # Combine all stop words
        all_stops = stop_words.union(spacy_stops).union(custom_stops)
        return all_stops

    def preprocess_text(self, text: str, remove_citations: bool = True) -> str:
        """Preprocess text with advanced stop word removal and cleaning."""
        try:
            # Convert to lowercase
            text = text.lower()

            # Remove citation patterns if requested
            if remove_citations:
                # Remove common citation patterns like (Author, Year) or [1]
                text = re.sub(r'\([A-Za-z]+,?\s+\d{4}\)', '', text)
                text = re.sub(r'\[\d+\]', '', text)

            # Tokenize text
            doc = self.nlp(text)

            # Remove stop words, punctuation, and numbers while preserving sentence structure
            processed_tokens = []
            for token in doc:
                if (
                        not token.is_stop
                        and not token.is_punct
                        and not token.like_num
                        and str(token).lower() not in self.stop_words
                        and len(str(token)) > 1  # Remove single characters
                ):
                    processed_tokens.append(str(token))

            # Reconstruct text while preserving some structure
            processed_text = ' '.join(processed_tokens)

            # Clean up extra whitespace
            processed_text = re.sub(r'\s+', ' ', processed_text).strip()

            return processed_text

        except Exception as e:
            logging.error(f"Error in text preprocessing: {str(e)}")
            return text  # Return original text if preprocessing fails

    def remove_boilerplate(self, text: str) -> str:
        """Remove common academic document boilerplate text."""
        # Patterns for common boilerplate text
        boilerplate_patterns = [
            r'all rights reserved',
            r'©.*?\d{4}',
            r'please cite as',
            r'this is a preprint',
            r'manuscript draft',
            r'do not distribute',
            r'confidential document',
            r'accepted for publication',
            r'under review',
            r'to appear in',
        ]

        for pattern in boilerplate_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)

        return text.strip()


class FileProcessor:
    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor()
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)

    def _generate_cache_path(self, file_path: Path) -> Path:
        """Generate a unique cache file path based on the input file."""
        # Create a hash of the file path and last modified time
        file_stat = file_path.stat()
        hash_input = f"{file_path.absolute()}{file_stat.st_mtime}"
        file_hash = hashlib.md5(hash_input.encode()).hexdigest()

        # Create cache file path
        return self.cache_dir / f"{file_hash}.txt"

    def _check_cache(self, file_path: Path) -> Optional[str]:
        """Check if a cached version exists and is valid."""
        cache_path = self._generate_cache_path(file_path)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_path}: {str(e)}")
                return None
        return None

    def _save_to_cache(self, file_path: Path, content: str) -> None:
        """Save processed content to cache."""
        cache_path = self._generate_cache_path(file_path)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            self.logger.warning(f"Error writing cache file {cache_path}: {str(e)}")

    def extract_pdf_content(self, file_path: Path) -> Optional[str]:
        """Extract text content from PDF with caching."""
        # Check cache first
        cached_content = self._check_cache(file_path)
        if cached_content is not None:
            self.logger.info(f"Using cached version of {file_path}")
            return cached_content

        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()

                text = self.preprocessor.remove_boilerplate(text)
                text = self.preprocessor.preprocess_text(text)

                # Save to cache
                self._save_to_cache(file_path, text)
                return text
        except Exception as e:
            self.logger.error(f"Error extracting PDF content from {file_path}: {str(e)}")
            return None

    def extract_docx_content(self, file_path: Path) -> Optional[str]:
        """Extract text content from DOCX with caching."""
        # Check cache first
        cached_content = self._check_cache(file_path)
        if cached_content is not None:
            self.logger.info(f"Using cached version of {file_path}")
            return cached_content

        try:
            text = docx2txt.process(file_path)
            text = self.preprocessor.remove_boilerplate(text)
            text = self.preprocessor.preprocess_text(text)

            # Save to cache
            self._save_to_cache(file_path, text)
            return text
        except Exception as e:
            self.logger.error(f"Error extracting DOCX content from {file_path}: {str(e)}")
            return None

    def create_document(self, file_path: Path) -> Optional[Document]:
        """Create a Document object from a file with caching."""
        try:
            # Extract content based on file type
            if file_path.suffix.lower() == '.pdf':
                content = self.extract_pdf_content(file_path)
            elif file_path.suffix.lower() == '.docx':
                content = self.extract_docx_content(file_path)
            else:
                self.logger.warning(f"Unsupported file type: {file_path}")
                return None

            if content:
                document = Document(
                    text=content,
                    metadata={
                        'file_name': file_path.name,
                        'file_path': str(file_path),
                        'file_type': file_path.suffix.lower()[1:],  # Remove the dot
                        'cache_path': str(self._generate_cache_path(file_path))
                    }
                )
                return document
            return None

        except Exception as e:
            self.logger.error(f"Error creating document from {file_path}: {str(e)}")
            return None

    def process_directory(self, directory: Path) -> List[Document]:
        """Process all files in directory and create Document objects."""
        self.logger.info(f"Processing directory: {directory}")

        # Collect all PDF and DOCX files
        files_to_process = []
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.docx']:
                files_to_process.append(file_path)

        if not files_to_process:
            self.logger.warning(f"No PDF or DOCX files found in {directory}")
            return []

        # Create Document objects directly from files
        documents = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.create_document, f)
                for f in files_to_process
            ]

            for future in tqdm(futures, desc="Processing documents"):
                result = future.result()
                if result:
                    documents.append(result)

        self.logger.info(f"Successfully processed {len(documents)} documents")
        return documents

    def clear_cache(self) -> None:
        """Clear all cached files."""
        try:
            for cache_file in self.cache_dir.glob('*.txt'):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")