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
        self.llm = LlamaIndexOllama(
            model=config.llm_model,
            temperature=config.temperature
        )
        self.base_query_engine = CitationQueryEngine.from_args(
            self.index,
            self.llm,
            similarity_top_k=config.k_retrieve,
            citation_chunk_size=config.chunk_size
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

    def get_file_path(self, node: Any) -> str:
        """Extract file path from node metadata."""
        try:
            metadata_str = node.node.get_metadata_str()
            return str(Path(metadata_str))
        except (AttributeError, TypeError):
            return "unknown_source"

    def validate_and_fix_citations(self, text: str, valid_source_count: int) -> str:
        """Strictly validate citations and fix or remove invalid ones."""
        citation_pattern = r'Source (\d+)'
        fixed_text = text
        position = 0

        while True:
            match = re.search(citation_pattern, fixed_text[position:])
            if not match:
                break

            start_idx = position + match.start()
            end_idx = position + match.end()
            source_num = int(match.group(1))

            # If citation number is invalid, remove the citation
            if source_num > valid_source_count:
                # Find the containing sentence or parenthetical expression
                text_before = fixed_text[:start_idx]
                text_after = fixed_text[end_idx:]

                # Remove the entire parenthetical if it exists
                paren_match = re.search(r'\([^)]*Source \d+[^)]*\)', fixed_text[max(0, start_idx - 50):end_idx + 50])
                if paren_match:
                    relative_start = max(0, start_idx - 50) + paren_match.start()
                    relative_end = max(0, start_idx - 50) + paren_match.end()
                    fixed_text = fixed_text[:relative_start] + fixed_text[relative_end:]
                    position = relative_start
                else:
                    # If not in parentheses, just remove the "Source X" text
                    fixed_text = text_before + text_after
                    position = start_idx
            else:
                position = end_idx

        return fixed_text.strip()

    def process_query(self, question: str, verbose: bool = False) -> Dict[str, Any]:
        """Process a single query and return results with strict citation validation."""
        start_time = time.time()

        # Get response from base query engine
        response = self.base_query_engine.query(self.config.guide + question)

        # Process sources first
        sources = []
        for i, source_node in enumerate(response.source_nodes, 1):
            source = {
                'id': i,
                'file': self.get_file_path(source_node),
                'score': getattr(source_node, 'score', None)
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