from dataclasses import dataclass
import yaml
import torch
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
    embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1"
    guide: str = "You are a research assistant helping analyze a set of interviews using thematic analysis."

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'MonkeyConfig':
        try:
            with open(yaml_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except FileNotFoundError:
            return cls()
        except Exception as e:
            print(f"Error loading config: {e}")
            return cls()

    def initialize_settings(self):
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=self.embedding_model,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        Settings.text_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
