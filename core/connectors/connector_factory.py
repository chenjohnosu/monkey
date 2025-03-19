"""
Factory for creating and managing connectors
"""

from typing import Dict, Any, Optional
from core.engine.utils import debug_print

class ConnectorFactory:
    """Factory class for creating and managing connectors"""
    
    def __init__(self, config):
        """Initialize the connector factory"""
        self.config = config
        self.connectors = {}
        debug_print(config, "Connector factory initialized")
        
    def get_vector_store_connector(self) -> Any:
        """
        Get the configured vector store connector
        
        Returns:
            The vector store connector instance
        """
        # Determine which vector store to use based on config
        vector_store_type = self.config.get('storage.vector_store', 'chroma')
        
        if vector_store_type == 'llama_index':
            # Use LlamaIndex connector
            if 'llama_index' not in self.connectors:
                from core.connectors.llama_index_connector import LlamaIndexConnector
                self.connectors['llama_index'] = LlamaIndexConnector(self.config)
            return self.connectors['llama_index']
            
        elif vector_store_type == 'haystack':
            # Use Haystack connector
            if 'haystack' not in self.connectors:
                from core.connectors.haystack_connector import HaystackConnector
                self.connectors['haystack'] = HaystackConnector(self.config)
            return self.connectors['haystack']
            
        else:
            # Use default built-in vector store
            if 'default' not in self.connectors:
                from core.storage import StorageManager
                self.connectors['default'] = StorageManager(self.config)
            return self.connectors['default']
            
    def get_llm_connector(self) -> Any:
        """
        Get the configured LLM connector
        
        Returns:
            The LLM connector instance
        """
        # Check if we should use Ollama
        llm_source = self.config.get('llm.source', 'ollama')
        
        if llm_source == 'ollama':
            # Use Ollama connector
            if 'ollama' not in self.connectors:
                from core.connectors.ollama_connector import OllamaConnector
                self.connectors['ollama'] = OllamaConnector(self.config)
            return self.connectors['ollama']
            
        elif llm_source in ('llama_index', 'llamaindex'):
            # Use LlamaIndex for LLM as well
            if 'llama_index_llm' not in self.connectors:
                # Lazy import to avoid circular imports
                debug_print(self.config, "Creating LlamaIndex LLM connector")
                
                # Create a specialized LLM connector using LlamaIndex
                try:
                    from llama_index.llms.ollama import OllamaLLM
                    from llama_index.core.llms import LLM

                    # Get default model from config
                    model = self.config.get('llm.default_model')

                    # Configure Ollama connection
                    host = self.config.get('llm.ollama_host')
                    port = self.config.get('llm.ollama_port')
                    base_url = f"{host}:{port}"

                    # Create LLM instance
                    llm = OllamaLLM(
                        model=model,
                        base_url=base_url,
                        request_timeout=120.0
                    )

                    # Store LLM instance for reuse
                    self.connectors['llama_index_llm'] = llm
                except Exception as e:
                    debug_print(self.config, f"Error creating LlamaIndex LLM connector: {str(e)}")
                    # Fall back to Ollama connector
                    return self.get_llm_connector_by_name('ollama')

            return self.connectors['llama_index_llm']

        else:
            # Fall back to Ollama connector
            return self.get_llm_connector_by_name('ollama')

    def get_llm_connector_by_name(self, name: str) -> Any:
        """
        Get a specific LLM connector by name

        Args:
            name (str): Connector name

        Returns:
            The LLM connector instance
        """
        if name == 'ollama':
            if 'ollama' not in self.connectors:
                from core.llm_connectors.ollama_connector import OllamaConnector
                self.connectors['ollama'] = OllamaConnector(self.config)
            return self.connectors['ollama']

        elif name == 'llama_index':
            if 'llama_index_llm' not in self.connectors:
                # Try to get LlamaIndex LLM
                return self.get_llm_connector()
            return self.connectors['llama_index_llm']

        else:
            debug_print(self.config, f"Unknown LLM connector name: {name}")
            return None

    def get_embedding_model(self, model_name=None):
        """
        Get a local embedding model

        Args:
            model_name (str, optional): Specific model name to use

        Returns:
            Embedding model instance
        """
        try:
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding

            # Use provided model name or get from config
            if not model_name:
                model_name = self.config.get('embedding.default_model', 'multilingual-e5')

            # Select appropriate local embedding model
            if model_name == "multilingual-e5":
                model_path = "intfloat/multilingual-e5-large"
            elif model_name == "mixbread":
                model_path = "mixedbread-ai/mxbai-embed-large-v1"
            elif model_name == "bge":
                model_path = "BAAI/bge-m3"
            else:
                # Fallback to a reliable multilingual model
                model_path = "intfloat/multilingual-e5-large"

            # Create and return local embedding model
            return HuggingFaceEmbedding(model_name_or_path=model_path)

        except Exception as e:
            debug_print(self.config, f"Error getting embedding model: {str(e)}")
            import traceback
            debug_print(self.config, traceback.format_exc())
            return None