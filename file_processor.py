# Standard library imports
import os
import re
import logging
import hashlib
from pathlib import Path
from typing import List, Optional, Dict
from concurrent.futures import ThreadPoolExecutor
from string import punctuation

# Third-party library imports
import PyPDF2
import docx2txt
import nltk
import spacy
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# LLamaIndex imports
from llama_index.core import Document

# Project-specific imports
from config import MonkeyConfig

# Optional OCR-related imports (install with pip)
try:
    import pytesseract
    import pdf2image
    from PIL import Image

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not installed. PDF OCR functionality will be limited.")


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


class PDFTextExtractor:
    """
    Advanced PDF text extraction with OCR fallback for image-based PDFs.

    Requirements:
    - pytesseract (pip install pytesseract)
    - pdf2image (pip install pdf2image)
    - Tesseract OCR installed on the system
    - poppler (for pdf2image conversion)
    """

    def __init__(self,
                 tesseract_path: Optional[str] = None,
                 ocr_languages: str = 'eng',
                 min_confidence: float = 50.0):
        """
        Initialize PDF text extractor with OCR capabilities.

        Args:
            tesseract_path (Optional[str]): Path to Tesseract executable
            ocr_languages (str): Language(s) for OCR (default: English)
            min_confidence (float): Minimum confidence threshold for OCR (0-100)
        """
        self.logger = logging.getLogger(__name__)

        # Set Tesseract path if provided
        if tesseract_path and OCR_AVAILABLE:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path

        self.ocr_languages = ocr_languages
        self.min_confidence = min_confidence

    def _is_pdf_image_based(self, pdf_path: str) -> bool:
        """
        Detect if PDF is primarily image-based by sampling first few pages.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            bool: True if PDF appears to be image-based, False otherwise
        """
        if not OCR_AVAILABLE:
            return False

        try:
            # Convert first few pages to images
            images = pdf2image.convert_from_path(
                pdf_path,
                first_page=1,
                last_page=min(3, self._get_pdf_page_count(pdf_path))
            )

            # Check if pages can be converted to text
            text_extraction_results = []
            for img in images:
                # Try extracting text from PDF first
                img_text = pytesseract.image_to_string(
                    img,
                    lang=self.ocr_languages,
                    config='--psm 6'
                )
                text_extraction_results.append(bool(img_text.strip()))

            # Consider PDF image-based if most sampled pages yield no direct text
            return sum(text_extraction_results) / len(text_extraction_results) < 0.5

        except Exception as e:
            self.logger.warning(f"Error detecting PDF type: {e}")
            return False

    def _get_pdf_page_count(self, pdf_path: str) -> int:
        """
        Get total number of pages in PDF.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            int: Number of pages in PDF
        """
        if not OCR_AVAILABLE:
            return 3

        try:
            return len(pdf2image.convert_from_path(pdf_path))
        except Exception as e:
            self.logger.warning(f"Error counting PDF pages: {e}")
            return 3  # Default to 3 pages if count fails

    def extract_text_with_ocr(self, pdf_path: str) -> Optional[str]:
        """
        Extract text from PDF, using OCR for image-based PDFs.

        Args:
            pdf_path (str): Path to PDF file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        if not OCR_AVAILABLE:
            return None

        try:
            # Detect if PDF is image-based
            if not self._is_pdf_image_based(pdf_path):
                return None  # Fallback to standard extraction

            self.logger.info(f"Performing OCR on image-based PDF: {pdf_path}")

            # Convert entire PDF to images
            images = pdf2image.convert_from_path(pdf_path)

            # Extract text from images with confidence tracking
            extracted_texts = []
            for img in images:
                # Perform OCR with detailed configuration
                ocr_result = pytesseract.image_to_data(
                    img,
                    lang=self.ocr_languages,
                    output_type=pytesseract.Output.DICT,
                    config='--psm 6'
                )

                # Filter text based on confidence
                valid_text_indices = [
                    i for i in range(len(ocr_result['text']))
                    if (float(ocr_result['conf'][i]) >= self.min_confidence
                        and ocr_result['text'][i].strip())
                ]

                # Combine high-confidence words
                page_text = ' '.join([
                    ocr_result['text'][i] for i in valid_text_indices
                ])

                extracted_texts.append(page_text)

            # Combine text from all pages
            full_text = '\n'.join(extracted_texts)

            return full_text if full_text.strip() else None

        except Exception as e:
            self.logger.error(f"OCR extraction failed for {pdf_path}: {e}")
            return None


class FileProcessor:
    def __init__(self, config: MonkeyConfig):
        """
        Initialize the FileProcessor with configuration and preprocessing tools.

        Args:
            config (MonkeyConfig): Configuration object for processing
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessor = TextPreprocessor()
        self.cache_dir = Path('.cache')
        self.cache_dir.mkdir(exist_ok=True)

        # Initialize OCR extractor if available
        if OCR_AVAILABLE:
            self.ocr_extractor = PDFTextExtractor(
                ocr_languages='eng',  # Can add multiple languages like 'eng+fra'
                min_confidence=60.0
            )
        else:
            self.ocr_extractor = None

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
            # Only save if content is not empty after stripping
            if content and content.strip():
                with open(cache_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.logger.info(f"Cached content for {file_path} successfully")
            else:
                self.logger.warning(f"No content to cache for {file_path}")
                # Optionally, remove the cache file if it exists
                if cache_path.exists():
                    cache_path.unlink()
        except Exception as e:
            self.logger.error(f"Error writing cache file {cache_path}: {str(e)}")

    def extract_pdf_content(self, file_path: Path) -> Optional[str]:
        """
        Extract text from PDF with optional OCR fallback.

        Args:
            file_path (Path): Path to PDF file

        Returns:
            Optional[str]: Extracted text or None if extraction fails
        """
        # Check cache first
        cached_content = self._check_cache(file_path)
        if cached_content is not None:
            self.logger.info(f"Using cached version of {file_path}")
            return cached_content

        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)

                # First, try standard text extraction
                text = ''
                for page in reader.pages:
                    text += page.extract_text()

                # If no text extracted and OCR is available, attempt OCR
                if (not text.strip() and self.ocr_extractor):
                    ocr_text = self.ocr_extractor.extract_text_with_ocr(str(file_path))
                    if ocr_text:
                        text = ocr_text

                # Process extracted text
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

            # Log detailed information about content extraction
            if not content:
                self.logger.warning(f"No content extracted from {file_path}")
                return None

            if not content.strip():
                self.logger.warning(f"Extracted content is empty after preprocessing for {file_path}")
                return None

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

        # Track files without content
        files_without_content = []

        # Create Document objects directly from files
        documents = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self.create_document, f)
                for f in files_to_process
            ]

            for future, file_path in zip(futures, files_to_process):
                result = future.result()
                if result:
                    documents.append(result)
                else:
                    files_without_content.append(file_path)

        # Log summary of processing
        self.logger.info(f"Successfully processed {len(documents)} documents")

        # If there are files without content, log them
        if files_without_content:
            print("\n=== Files Without Processable Content ===")
            for file in files_without_content:
                print(f"- {file}")
            print(f"Total files without content: {len(files_without_content)}\n")

        return documents

    def clear_cache(self) -> None:
        """Clear all cached files."""
        try:
            for cache_file in self.cache_dir.glob('*.txt'):
                cache_file.unlink()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")