from dataclasses import dataclass
import yaml
import torch
import warnings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter


@dataclass
class MonkeyConfig:
    src_dir: str = "src"
    vdb_dir: str = "vdb"
    llm_model: str = "mistral"
    temperature: float = 0.7
    k_retrieve: int = 5
    line_width: int = 80
    chunk_size: int = 1024
    chunk_overlap: int = 200
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"  # Changed from embedding_model_name to embedding_model
    # ORIG guide: str = "You are a very intelligent text wrangler and researcher."
    guide: str = "I am a qualitative academic researcher helping with analyzing a series of interviews and completing thematic analysis. The text will be in chinese and the analysis should be done in chinese then translated into english; keep original text with english translations. There will be lots of repetitive questions that will be asked to each respondent: ignore these"

    def __post_init__(self):
        # Suppress specific PyTorch flash attention warning
        warnings.filterwarnings("ignore", message=".*Torch was not compiled with flash attention.*")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MonkeyConfig':
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)

            # Create a default instance first to get default values
            default_config = cls()

            # If config_dict is None (empty YAML file), use empty dict
            if config_dict is None:
                config_dict = {}

            # Make sure embedding_model is included
            if 'embedding_model' not in config_dict:
                config_dict['embedding_model'] = default_config.embedding_model
                print(f"Adding missing 'embedding_model' with default: {default_config.embedding_model}")

            # Create new instance with the merged configuration
            return cls(**config_dict)
        except FileNotFoundError:
            return cls()
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()

    def initialize_settings(self):
        """Initialize settings with optimized CUDA configuration."""
        # Force CUDA initialization if available
        if torch.cuda.is_available():
            print("\nOptimizing GPU settings for maximum performance...")
            torch.cuda.init()

            # More aggressive GPU memory optimization
            torch.cuda.empty_cache()

            # Set optimal tensor math settings
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True

            device = "cuda"
        else:
            device = "cpu"

        # Initialize the embedding model with optimized settings
        # Removed deprecated parameters: pooling, cache_folder, embed_batch_size
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model,  # This line matches the attribute name
            device=device,
            model_kwargs={
                "device_map": "auto",
                "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
                "trust_remote_code": True
            }
        )

        # Initialize text splitter
        Settings.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

        # Verify and log GPU configuration
        if torch.cuda.is_available():
            print(f"\nGPU Configuration:")
            print(f"Device: {device}")
            print(f"GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"Initial GPU Memory: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")

            # Generate a test embedding to verify GPU usage
            try:
                test_text = "GPU test embedding"
                _ = Settings.embed_model.get_text_embedding(test_text)
                print(f"GPU Memory After Test: {torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB")
                print("✓ GPU successfully initialized and tested")
            except Exception as e:
                print(f"Error testing GPU: {str(e)}")