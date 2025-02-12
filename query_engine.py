from typing import Any, Dict, List, Optional
import time
import textwrap
from pathlib import Path
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from config import MonkeyConfig


class QueryEngine:
    def __init__(self, config: MonkeyConfig, index: Any):
        self.config = config
        self.index = index
        self.llm = LlamaIndexOllama(
            model=config.llm_model,
            temperature=config.temperature
        )
        self.query_engine = CitationQueryEngine.from_args(
            self.index,
            self.llm,
            similarity_top_k=config.k_retrieve,
            citation_chunk_size=config.chunk_size
        )

    def process_query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """Process a single query and return results."""
        start_time = time.time()
        response = self.query_engine.query(self.config.guide + question)
        elapsed_time = time.time() - start_time

        wrapped_response = textwrap.wrap(
            response.response,
            self.config.line_width
        )

        sources = []
        for i, source_node in enumerate(response.source_nodes):
            metadata_str = source_node.node.get_metadata_str()
            file_path = Path(metadata_str)

            source = {
                'id': i + 1,
                'file': str(file_path),  # Full path
                'filename': file_path.name,  # Just the filename
                'stem': file_path.stem,  # Filename without extension
                'extension': file_path.suffix  # File extension
            }

            if verbose:
                source['text'] = source_node.node.get_text()
                source['score'] = source_node.score if hasattr(source_node, 'score') else None

            sources.append(source)

        return {
            'response': wrapped_response,
            'sources': sources,
            'elapsed_time': elapsed_time
        }