import os
import re
import logging
import PyPDF2
import docx2txt
from pathlib import Path
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from string import punctuation
import spacy

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

    def _get_enhanced_stopwords(self) -> Set[str]:
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
        """
        Preprocess text with advanced stop word removal and cleaning.
        """
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

    def clean_text(self, text: str) -> str:
        """Clean up and preprocess extracted text from documents."""
        # Basic cleaning
        text = text.replace('\n', ' ')
        text = re.sub(r'\s+', ' ', text)

        # Remove boilerplate
        text = self.preprocessor.remove_boilerplate(text)

        # Advanced preprocessing including stop word removal
        text = self.preprocessor.preprocess_text(text)

        return text.strip()

    def convert_single_file(self, file_path: Path) -> Optional[Path]:
        """Convert a single file to text format."""
        try:
            if file_path.suffix == '.pdf':
                return self._convert_pdf(file_path)
            elif file_path.suffix == '.docx':
                return self._convert_docx(file_path)
            return None
        except Exception as e:
            self.logger.error(f"Error converting {file_path}: {str(e)}")
            return None

    def _convert_pdf(self, file_path: Path) -> Optional[Path]:
        """Convert PDF to text."""
        output_path = file_path.with_suffix('.txt')
        if output_path.exists():
            self.logger.info(f"Skipping {output_path}, already exists")
            return output_path

        try:
            with open(file_path, 'rb') as pdf_file:
                reader = PyPDF2.PdfReader(pdf_file)
                text = ''
                for page in reader.pages:
                    text += page.extract_text()

            text = self.clean_text(text)
            with open(output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            return output_path
        except Exception as e:
            self.logger.error(f"Error converting PDF {file_path}: {str(e)}")
            return None

    def _convert_docx(self, file_path: Path) -> Optional[Path]:
        """Convert DOCX to text."""
        output_path = file_path.with_suffix('.txt')
        if output_path.exists():
            self.logger.info(f"Skipping {output_path}, already exists")
            return output_path

        try:
            text = docx2txt.process(file_path)
            text = self.clean_text(text)
            with open(output_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text)
            return output_path
        except Exception as e:
            self.logger.error(f"Error converting DOCX {file_path}: {str(e)}")
            return None

    def process_directory(self, directory: Path) -> List[Path]:
        """Process all files in directory and convert to text."""
        self.logger.info(f"Processing directory: {directory}")
        files_to_convert = []
        for file_path in directory.rglob('*'):
            if file_path.suffix.lower() in ['.pdf', '.docx']:
                files_to_convert.append(file_path)

        converted_files = []
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.convert_single_file, f) for f in files_to_convert]
            for future in tqdm(futures, desc="Converting files"):
                result = future.result()
                if result:
                    converted_files.append(result)

        return converted_files