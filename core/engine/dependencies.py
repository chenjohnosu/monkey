"""
Centralized dependency checking and management
"""

import importlib.util
import sys
from typing import Dict, List, Optional
from core.engine.logging import debug, info, warning, error


class DependencyManager:
    """Centralized dependency management for optional libraries"""

    def __init__(self):
        """Initialize the dependency manager with empty caches"""
        self._status_cache = {}  # Cache of dependency statuses
        self._model_cache = {}  # Cache of loaded models

        # Set of already issued warnings to avoid duplicates
        self._warned = set()

        # Core dependencies that should be present
        self.core_dependencies = {
            'yaml': 'PyYAML',
            'numpy': 'numpy',
            're': None,  # Built-in
            'os': None,  # Built-in
            'json': None  # Built-in
        }

        # Optional dependencies and their pip package names
        self.optional_dependencies = {
            # Text processing
            'jieba': 'jieba',
            'nltk': 'nltk',
            'spacy': 'spacy',
            'spacy_model_en': 'spacy>=3.0.0,<4.0.0',
            'spacy_model_zh': 'spacy[zh]>=3.0.0,<4.0.0',

            # Machine learning
            'sklearn': 'scikit-learn',
            'transformers': 'transformers',
            'torch': 'torch',

            # Data processing
            'pandas': 'pandas',

            # Plotting/visualization
            'matplotlib': 'matplotlib',
            'plotly': 'plotly',

            # Document processing
            'docx': 'python-docx',
            'PyPDF2': 'PyPDF2',
            'pdfminer': 'pdfminer.six',

            # LLM and vector stores
            'llama_index.core': 'llama-index',
            'haystack': 'haystack-ai',
            'chromadb': 'chromadb',

            # Dimensionality reduction and clustering
            'umap': 'umap-learn',
            'hdbscan': 'hdbscan',
            'networkx': 'networkx',
            'community': 'python-louvain'
        }

        # Initialize status cache
        self._check_core_dependencies()

    def _check_core_dependencies(self):
        """Check and log the status of core dependencies"""
        info("Checking core dependencies...")

        for module_name, package_name in self.core_dependencies.items():
            if package_name is None:  # Built-in module
                if module_name in sys.modules:
                    debug(f"Core module {module_name} is available (built-in)")
                    self._status_cache[module_name] = True
                else:
                    error(f"Core built-in module {module_name} is unexpectedly unavailable!")
                    self._status_cache[module_name] = False
            else:
                is_available = self.is_available(module_name)
                if is_available:
                    debug(f"Core dependency {module_name} is available")
                else:
                    error(f"Core dependency {module_name} is missing! System may not function correctly.")

    def is_available(self, module_name: str) -> bool:
        """
        Check if a module is available

        Args:
            module_name: The name of the module to check

        Returns:
            bool: True if available, False otherwise
        """
        # Return cached result if available
        if module_name in self._status_cache:
            return self._status_cache[module_name]

        # Check if module is available
        is_available = False
        try:
            spec = importlib.util.find_spec(module_name)
            is_available = spec is not None
        except (ModuleNotFoundError, ImportError, ValueError):
            is_available = False

        # Cache and return result
        self._status_cache[module_name] = is_available
        return is_available

    def require(self, module_name: str, feature_name: Optional[str] = None) -> bool:
        """
        Check for required dependency with user-friendly error message

        Args:
            module_name: The name of the module to check
            feature_name: Optional name of the feature requiring this dependency

        Returns:
            bool: True if available, False otherwise
        """
        available = self.is_available(module_name)

        if not available:
            package_name = self.optional_dependencies.get(module_name, module_name)

            # Create warning message
            if feature_name:
                message = f"'{module_name}' is required for {feature_name} but not installed."
            else:
                message = f"Required dependency '{module_name}' is not installed."

            # Add installation instructions
            if package_name:
                message += f" Install with: pip install {package_name}"

            # Only warn once per module
            warning_key = f"{module_name}_{feature_name}"
            if warning_key not in self._warned:
                warning(message)
                self._warned.add(warning_key)

        return available

    def check_spacy_model(self, model_name: str) -> bool:
        """
        Check if a spaCy model is installed

        Args:
            model_name: Name of the spaCy model to check

        Returns:
            bool: True if available, False otherwise
        """
        if not self.require('spacy', 'spaCy language model'):
            return False

        try:
            import spacy

            # Check if model is installed
            try:
                spacy.load(model_name)
                return True
            except OSError:
                return False
        except Exception as e:
            warning(f"Error checking spaCy model {model_name}: {str(e)}")
            return False

    def get_available_spacy_models(self) -> Dict[str, List[str]]:
        """
        Get list of available spaCy models

        Returns:
            Dict: Dictionary mapping language codes to available models
        """
        if not self.require('spacy', 'spaCy language models'):
            return {}

        try:
            import spacy
            from spacy.cli.info import info

            # Get installed models
            installed_models = {}
            model_info = info(False)

            if 'pipelines' in model_info:
                for model_name in model_info['pipelines']:
                    # Extract language code from model name
                    lang_code = model_name.split('_')[0]

                    if lang_code not in installed_models:
                        installed_models[lang_code] = []

                    installed_models[lang_code].append(model_name)

            return installed_models
        except Exception as e:
            warning(f"Error getting available spaCy models: {str(e)}")
            return {}

    def ensure_nltk_data(self, data_name: str) -> bool:
        """
        Ensure NLTK data is downloaded

        Args:
            data_name: The name of the NLTK data to check/download

        Returns:
            bool: True if available, False otherwise
        """
        if not self.require('nltk', 'NLTK data download'):
            return False

        try:
            import nltk
            # First ensure nltk.data is initialized
            # This is a critical step to avoid the 'module nltk has no attribute data' error
            nltk.download('punkt', quiet=True)  # Download a basic dataset to initialize nltk.data

            try:
                nltk.data.find(data_name)
                return True
            except LookupError:
                info(f"Downloading NLTK {data_name}...")
                nltk.download(data_name, quiet=True)
                return True
        except Exception as e:
            warning(f"Error ensuring NLTK data {data_name}: {str(e)}")
            return False

    def check_required_modules(self, module_list: List[str], feature_name: str) -> bool:
        """
        Check if all modules in a list are available

        Args:
            module_list: List of module names to check
            feature_name: Name of the feature requiring these modules

        Returns:
            bool: True if all available, False otherwise
        """
        all_available = True
        for module in module_list:
            if not self.require(module, feature_name):
                all_available = False

        return all_available

    def get_model_alternatives(self, model_type: str) -> List[str]:
        """
        Get available alternative models for a given type

        Args:
            model_type: Type of model (embedding, llm, etc.)

        Returns:
            List of available model options
        """
        if model_type == 'embedding':
            alternatives = []

            # Check for various embedding models
            if self.is_available('transformers') and self.is_available('torch'):
                alternatives.extend(['multilingual-e5', 'bge', 'mixbread'])

            if self.is_available('transformers') and self.is_available('torch') and self.is_available('jieba'):
                alternatives.append('jina-zh')

            # Add spaCy models if available
            if self.is_available('spacy'):
                try:
                    spacy_models = self.get_available_spacy_models()
                    if 'en' in spacy_models:
                        alternatives.append('spacy_en')
                    if 'zh' in spacy_models:
                        alternatives.append('spacy_zh')
                except:
                    pass

            return alternatives

        elif model_type == 'llm':
            alternatives = []

            # Check for various LLM backends
            if self.is_available('transformers') and self.is_available('torch'):
                alternatives.extend(['llama2', 'mistral'])

            if self.is_available('ollama'):
                alternatives.extend(['ollama'])

            return alternatives

        return []

    def check_hardware_support(self) -> Dict[str, bool]:
        """
        Check available hardware acceleration support

        Returns:
            Dict with keys 'cuda', 'mps', etc. and boolean values
        """
        support = {
            'cuda': False,
            'mps': False,
            'cpu': True
        }

        # Check CUDA support
        if self.is_available('torch'):
            try:
                import torch
                support['cuda'] = torch.cuda.is_available()

                # Check for MPS (Apple Silicon)
                support['mps'] = (hasattr(torch.backends, 'mps') and
                                  torch.backends.mps.is_available())
            except Exception as e:
                warning(f"Error checking hardware support: {str(e)}")

        return support

    def get_status_report(self) -> Dict[str, Dict[str, bool]]:
        """
        Get a comprehensive report of all dependency statuses

        Returns:
            Dict with categories and module availability
        """
        report = {
            'core': {},
            'text_processing': {},
            'machine_learning': {},
            'document_processing': {},
            'visualization': {},
            'llm': {},
            'hardware': {}
        }

        # Fill in core dependencies
        for module in self.core_dependencies:
            report['core'][module] = self.is_available(module)

        # Text processing
        text_modules = ['jieba', 'nltk', 'spacy']
        for module in text_modules:
            report['text_processing'][module] = self.is_available(module)

        # Add spaCy models if spaCy is available
        if self.is_available('spacy'):
            report['text_processing']['spacy_en_model'] = self.check_spacy_model('en_core_web_sm')
            report['text_processing']['spacy_zh_model'] = self.check_spacy_model('zh_core_web_sm')

        # Machine learning
        ml_modules = ['sklearn', 'torch', 'transformers', 'umap', 'hdbscan', 'networkx', 'community']
        for module in ml_modules:
            report['machine_learning'][module] = self.is_available(module)

        # Document processing
        doc_modules = ['docx', 'PyPDF2', 'pdfminer']
        for module in doc_modules:
            report['document_processing'][module] = self.is_available(module)

        # Visualization
        viz_modules = ['matplotlib', 'plotly']
        for module in viz_modules:
            report['visualization'][module] = self.is_available(module)

        # LLM and vector stores
        llm_modules = ['llama_index.core', 'haystack', 'chromadb']
        for module in llm_modules:
            report['llm'][module] = self.is_available(module)

        # Hardware support
        report['hardware'] = self.check_hardware_support()

        return report

    def format_status_report(self) -> str:
        """Format dependency status report for display"""
        report = self.get_status_report()

        lines = ["Dependency Status Report:", ""]

        for category, modules in report.items():
            lines.append(f"=== {category.upper()} ===")

            if category == 'hardware':
                # Special formatting for hardware
                for device, available in modules.items():
                    status = "✓ Available" if available else "✗ Not available"
                    lines.append(f"  {device.upper()}: {status}")
            else:
                # Regular module formatting
                for module, available in modules.items():
                    status = "✓ Available" if available else "✗ Not available"
                    package = self.optional_dependencies.get(module, "")
                    if package and package != module:
                        lines.append(f"  {module} ({package}): {status}")
                    else:
                        lines.append(f"  {module}: {status}")

            lines.append("")

        return "\n".join(lines)


# Create singleton instance
dependency_manager = DependencyManager()


# Convenience functions to avoid importing the manager directly
def is_available(module_name: str) -> bool:
    """Check if a module is available"""
    return dependency_manager.is_available(module_name)


def require(module_name: str, feature_name: Optional[str] = None) -> bool:
    """Require a module with helpful error message"""
    return dependency_manager.require(module_name, feature_name)


def check_required_modules(module_list: List[str], feature_name: str) -> bool:
    """Check if all modules in a list are available"""
    return dependency_manager.check_required_modules(module_list, feature_name)


def ensure_nltk_data(data_name: str) -> bool:
    """Ensure NLTK data is downloaded"""
    return dependency_manager.ensure_nltk_data(data_name)


def get_status_report() -> Dict[str, Dict[str, bool]]:
    """Get dependency status report"""
    return dependency_manager.get_status_report()


def format_status_report() -> str:
    """Format dependency status report for display"""
    return dependency_manager.format_status_report()


def check_hardware_support() -> Dict[str, bool]:
    """Check available hardware acceleration support"""
    return dependency_manager.check_hardware_support()


def get_model_alternatives(model_type: str) -> List[str]:
    """Get available alternative models for a given type"""
    return dependency_manager.get_model_alternatives(model_type)


def check_spacy_model(model_name: str) -> bool:
    """Check if a spaCy model is installed"""
    return dependency_manager.check_spacy_model(model_name)


def get_available_spacy_models() -> Dict[str, List[str]]:
    """Get a dict of available spaCy models by language code"""
    return dependency_manager.get_available_spacy_models()