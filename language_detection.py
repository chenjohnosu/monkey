# Language Detection Utility
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Third-party library imports
import langdetect
import spacy
import jieba
import PyPDF2
import docx2txt

# Project-specific imports
from config import MonkeyConfig


class LanguageDetector:
    def __init__(self, src_dir: str):
        """
        Initialize language detector for a given source directory.

        Args:
            src_dir (str): Path to the source directory containing documents
        """
        self.src_dir = Path(src_dir)
        self.logger = logging.getLogger(__name__)

        # Load spaCy models for language-specific processing
        try:
            self.en_nlp = spacy.load('en_core_web_sm')
            self.zh_nlp = spacy.load('zh_core_web_sm')
        except OSError:
            self.logger.warning("SpaCy models not fully loaded. Some language detection may be limited.")
            self.en_nlp = None
            self.zh_nlp = None

    def _read_file_content(self, file_path: Path, max_chars: int = 5000) -> Optional[str]:
        """
        Read content from a file, handling different file types.

        Args:
            file_path (Path): Path to the file
            max_chars (int): Maximum number of characters to read

        Returns:
            Optional[str]: File content or None if reading fails
        """
        try:
            # Handle PDF files
            if file_path.suffix.lower() == '.pdf':
                with open(file_path, 'rb') as pdf_file:
                    reader = PyPDF2.PdfReader(pdf_file)
                    text = ''
                    for page in reader.pages[:3]:  # Read first 3 pages
                        text += page.extract_text()
                        if len(text) >= max_chars:
                            break
                return text[:max_chars]

            # Handle DOCX files
            elif file_path.suffix.lower() == '.docx':
                text = docx2txt.process(file_path)
                return text[:max_chars]

            # Handle TXT files
            elif file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read(max_chars)

            return None

        except Exception as e:
            self.logger.warning(f"Error reading {file_path}: {e}")
            return None

    def detect_document_languages(self) -> Dict[str, List[str]]:
        """
        Detect languages across all documents in the source directory.

        Returns:
            Dict containing language detection results
        """
        # Supported file types
        supported_extensions = ['.pdf', '.docx', '.txt']

        # Collect documents
        documents = []
        for ext in supported_extensions:
            documents.extend(list(self.src_dir.rglob(f'*{ext}'))[:50])  # Limit to first 50 docs

        # Language detection results
        results = {
            'detected_languages': [],
            'file_languages': {}
        }

        # Configure langdetect to include more languages
        langdetect.DetectorFactory.seed = 0  # For reproducibility

        # Detect language for each document
        for doc_path in documents:
            try:
                content = self._read_file_content(doc_path)
                if not content:
                    continue

                # Use langdetect for initial language detection
                try:
                    detected_lang = langdetect.detect(content)
                except Exception:
                    detected_lang = 'unknown'

                # Store results
                results['detected_languages'].append(detected_lang)
                results['file_languages'][doc_path.name] = detected_lang

            except Exception as e:
                self.logger.warning(f"Error processing {doc_path}: {e}")

        return results

    def analyze_language_characteristics(self) -> Dict[str, Any]:
        """
        Provide detailed language analysis for the document set.

        Returns:
            Dict with language-specific insights
        """
        lang_results = self.detect_document_languages()

        # Count language distributions
        lang_counts = {}
        for lang in lang_results['detected_languages']:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

        # Prepare detailed analysis
        analysis = {
            'total_documents': len(lang_results['detected_languages']),
            'language_distribution': {
                lang: count / len(lang_results['detected_languages']) * 100
                for lang, count in lang_counts.items()
            },
            'sample_languages': list(lang_counts.keys())
        }

        return analysis


def display_language_info(config: MonkeyConfig):
    """
    Display comprehensive language information for the document set.

    Args:
        config (MonkeyConfig): Configuration object
    """
    print("\nLanguage Analysis:")
    print("=" * 50)

    try:
        # Initialize language detector
        detector = LanguageDetector(config.src_dir)

        # Detect languages
        lang_analysis = detector.analyze_language_characteristics()

        print(f"Total Documents Sampled: {lang_analysis['total_documents']}")
        print("\nLanguage Distribution:")
        for lang, percentage in sorted(
                lang_analysis['language_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
        ):
            print(f"  - {lang}: {percentage:.2f}%")

        print("\nSample Languages Detected:",
              ", ".join(lang_analysis['sample_languages']))

        # Provide configuration context
        print("\nCurrent Configuration:")
        print(f"Configured Language: {config.language}")

        # Recommend language setting based on detection
        if lang_analysis['sample_languages']:
            dominant_lang = max(
                lang_analysis['language_distribution'],
                key=lang_analysis['language_distribution'].get
            )
            print("\nRecommendation:")
            if dominant_lang == 'zh-cn' and config.language != 'zh':
                print("🔍 Detected primarily Chinese documents. Consider setting language to 'zh'.")
            elif dominant_lang == 'en' and config.language != 'en':
                print("🔍 Detected primarily English documents. Consider setting language to 'en'.")

    except Exception as e:
        print(f"Error during language analysis: {e}")


# Mapping of language codes for more user-friendly display
LANGUAGE_NAMES = {
    'en': 'English',
    'zh-cn': 'Chinese (Simplified)',
    'zh-tw': 'Chinese (Traditional)',
    'unknown': 'Unknown'
}