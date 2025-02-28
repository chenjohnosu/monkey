from typing import Any, Dict, List
import time
import textwrap
import re
from pathlib import Path
import logging
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.llms.ollama import Ollama as LlamaIndexOllama
from config import MonkeyConfig


class QueryEngine:
    def __init__(self, config: MonkeyConfig, index: Any):
        self.config = config
        self.index = index
        self.logger = logging.getLogger(__name__)

        # Simple Ollama configuration with minimal parameters
        self.llm = LlamaIndexOllama(
            model=config.llm_model,
            temperature=config.temperature,
            request_timeout=120.0
        )

        # Configure basic query engine
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

    def process_query(self, question: str, verbose: bool = False, unique_sources: bool = False) -> Dict[str, Any]:
        """Process a single query with basic citation handling."""
        start_time = time.time()
        print(f"Processing query: {question[:50]}...")

        try:
            # Get response from base query engine
            print("Querying database...")
            response = self.base_query_engine.query(self.config.guide + question)
            print("Query complete, processing results...")

            # Initialize sources list
            sources = []

            # Skip processing if no source nodes
            if not hasattr(response, 'source_nodes') or len(response.source_nodes) == 0:
                print("No sources found.")
                return {
                    'response': ["No sources found for this query."],
                    'sources': [],
                    'elapsed_time': time.time() - start_time
                }

            # Handle unique sources mode
            if unique_sources:
                print("Processing unique sources...")
                # Group by source file
                files_dict = {}
                for i, node in enumerate(response.source_nodes):
                    file_name = self.get_file_name(node)
                    if file_name not in files_dict:
                        files_dict[file_name] = []
                    files_dict[file_name].append((i, node))

                # Take highest scoring node from each file
                best_nodes = []
                for file_nodes in files_dict.values():
                    best_node = max(file_nodes, key=lambda x: getattr(x[1], 'score', 0))
                    best_nodes.append(best_node)

                # Sort by score
                best_nodes.sort(key=lambda x: getattr(x[1], 'score', 0), reverse=True)

                # Limit to k nodes
                best_nodes = best_nodes[:self.config.k_retrieve]

                # Create new source nodes list and reorder
                new_nodes = [node for _, node in best_nodes]
                response.source_nodes = new_nodes

            # Process source nodes
            print(f"Processing {len(response.source_nodes)} source nodes...")
            for i, source_node in enumerate(response.source_nodes, 1):
                file_name = self.get_file_name(source_node)
                score = getattr(source_node, 'score', 0)

                source = {
                    'id': i,
                    'file': file_name,
                    'file_name': file_name,
                    'display': f"[{i}] {file_name} (score: {score:.3f})",
                    'score': score
                }
                if verbose:
                    print(f"Getting text for source {i}...")
                    source['text'] = self.get_node_text(source_node.node)
                sources.append(source)

            # Get the response text
            response_text = response.response

            # Clean up response text
            if "References:" in response_text:
                print("Removing references section...")
                response_text = response_text.split("References:")[0]

            # Basic citation cleanup - replace Source X with [X]
            print("Cleaning citations...")
            for i in range(len(sources), 0, -1):
                response_text = response_text.replace(f"Source {i}", f"[{i}]")

            # Wrap text for output
            print("Wrapping text...")
            wrapped_response = textwrap.wrap(response_text, self.config.line_width)

            print("Query processing completed.")
            return {
                'response': wrapped_response,
                'sources': sources,
                'elapsed_time': time.time() - start_time
            }

        except Exception as e:
            import traceback
            error_msg = f"Error processing query: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
            return {
                'response': [f"Error processing query: {str(e)}"],
                'sources': [],
                'elapsed_time': time.time() - start_time
            }


def display_response_diagnostics(result: dict, config: MonkeyConfig):
    """Display diagnostic information for a response."""
    print("\nResponse Diagnostics:")
    print("-" * 50)

    # Model Information
    print(f"Model: {config.llm_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Response Time: {result['elapsed_time']:.2f}s")

    # Processing Information
    print(f"Sources Retrieved: {len(result['sources'])}")

    # Source Details
    if result['sources']:
        print("\nSources:")
        for source in result['sources']:
            print(source['display'])