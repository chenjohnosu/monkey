from typing import Any, Dict, List, Tuple
import time
import textwrap
import re
from pathlib import Path
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from config import MonkeyConfig


class QueryEngine:
    def __init__(self, config: MonkeyConfig, index: Any):
        self.config = config
        self.index = index

        # Configure Ollama with optimal settings for GPU
        self.llm = LlamaIndexOllama(
            model=config.llm_model,
            temperature=config.temperature,
            request_timeout=120.0,
            context_window=4096,
            additional_kwargs={
                "numa": True,
                "f16": True,
                "batch_size": 512
            }
        )

        # Configure query engine with CUDA optimization
        self.base_query_engine = CitationQueryEngine.from_args(
            self.index,
            self.llm,
            similarity_top_k=config.k_retrieve,
            citation_chunk_size=config.chunk_size,
            use_async=True,
            show_progress=False
        )

    def get_node_text(self, node: Any) -> str:
        """Safely extract text from a node."""
        if hasattr(node, 'text_resource'):
            return node.text_resource
        if hasattr(node, 'get_content'):
            return node.get_content()
        if hasattr(node, 'text'):
            return node.text
        return ""

    def get_file_name(self, node: Any) -> str:
        """Extract just the filename from node metadata."""
        try:
            if hasattr(node.node, 'metadata'):
                metadata = node.node.metadata
                if 'file_name' in metadata:
                    return metadata['file_name']
                elif 'file_path' in metadata:
                    return Path(metadata['file_path']).name

            metadata_str = node.node.get_metadata_str()
            if 'file_name:' in metadata_str:
                return metadata_str.split('file_name:')[1].split('\n')[0].strip()
            return Path(metadata_str).name
        except (AttributeError, TypeError):
            return "unknown_source"

    def validate_and_fix_citations(self, text: str, valid_source_count: int) -> str:
        """Convert 'Source X' citations to '[X]' format and validate them."""
        # First, handle existing "[X]" format citations
        citation_pattern = r'Source (\d+)(?:\s*\[(?:\d+)\])?'
        fixed_text = text
        position = 0

        while True:
            match = re.search(citation_pattern, fixed_text[position:])
            if not match:
                break

            start_idx = position + match.start()
            end_idx = position + match.end()
            source_num = int(match.group(1))

            if source_num <= valid_source_count:
                # Replace "Source X" or "Source X [X]" with just "[X]"
                fixed_text = fixed_text[:start_idx] + f"[{source_num}]" + fixed_text[end_idx:]
                position = start_idx + len(f"[{source_num}]")
            else:
                # Remove invalid citations
                text_before = fixed_text[:start_idx]
                text_after = fixed_text[end_idx:]
                # Check for parenthetical citations
                paren_match = re.search(r'\([^)]*Source \d+[^)]*\)', fixed_text[max(0, start_idx - 50):end_idx + 50])
                if paren_match:
                    relative_start = max(0, start_idx - 50) + paren_match.start()
                    relative_end = max(0, start_idx - 50) + paren_match.end()
                    fixed_text = fixed_text[:relative_start] + fixed_text[relative_end:]
                    position = relative_start
                else:
                    fixed_text = text_before + text_after
                    position = start_idx

        return fixed_text.strip()

    def process_query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """Process a single query and return results with strict citation validation."""
        start_time = time.time()

        # Get response from base query engine
        response = self.base_query_engine.query(self.config.guide + question)

        # Process sources first
        sources = []
        for i, source_node in enumerate(response.source_nodes, 1):
            file_name = self.get_file_name(source_node)
            score = getattr(source_node, 'score', 0)

            source = {
                'id': i,
                'file': file_name,
                'file_name': file_name,
                'display': f"[{i}] {file_name} (score: {score:.3f})",  # Updated display format
                'score': score
            }
            if verbose:
                source['text'] = self.get_node_text(source_node.node)
            sources.append(source)

        # Clean and validate response text
        response_text = " ".join(textwrap.wrap(response.response, self.config.line_width))
        if "References:" in response_text:
            response_text = response_text.split("References:")[0]

        # Validate and fix citations
        validated_text = self.validate_and_fix_citations(response_text, len(sources))

        # Wrap the cleaned and validated text
        wrapped_response = textwrap.wrap(validated_text, self.config.line_width)

        return {
            'response': wrapped_response,
            'sources': sources,
            'elapsed_time': time.time() - start_time
        }


def display_response_diagnostics(result: dict, config: MonkeyConfig):
    """Display diagnostic information for a response."""
    from cuda_utils import CUDAChecker
    cuda_checker = CUDAChecker()

    print("\nResponse Diagnostics:")
    print("-" * 50)

    # Model Information
    print(f"Model: {config.llm_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Response Time: {result['elapsed_time']:.2f}s")

    # Processing Information
    print(f"Sources Retrieved: {len(result['sources'])}")
    print(f"Embedding Device: {cuda_checker.check_embedding_device()}")

    # Source Details
    if result['sources']:
        print("\nSources:")
        for source in result['sources']:
            print(source['display'])  # Already in [X] format from earlier