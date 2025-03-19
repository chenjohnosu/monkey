"""
Query mode module with LlamaIndex and Ollama integration
"""

import os
from core.engine.utils import debug_print
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.connectors.connector_factory import ConnectorFactory

class QueryEngine:
    """Interactive query engine using LlamaIndex and Ollama"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the query engine"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)
        self.factory = ConnectorFactory(config)
        self.llm_connector = self.factory.get_llm_connector()

        self.active = False
        self.current_workspace = None
        debug_print(config, "Query engine initialized")

    def activate(self, workspace):
        """
        Activate query mode for a workspace

        Args:
            workspace (str): The workspace to query
        """
        debug_print(self.config, f"Activating query mode for workspace '{workspace}'")

        # Check if workspace exists
        data_dir = os.path.join("data", workspace)
        if not os.path.exists(data_dir):
            print(f"Workspace '{workspace}' does not exist or has no vector store")
            return False

        # Set active state
        self.active = True
        self.current_workspace = workspace

        # Load vector store
        if not self.storage_manager.load_vector_store(workspace):
            print(f"Failed to load vector store for workspace '{workspace}'")
            self.active = False
            return False

        # Connect to Ollama and verify model availability
        if not self._verify_llm_connection():
            print("Warning: LLM connection could not be verified. Will attempt to use when needed.")

        return True

    def deactivate(self):
        """Deactivate query mode"""
        debug_print(self.config, "Deactivating query mode")

        # Store the workspace name for the message
        workspace = self.current_workspace

        # Reset state
        self.active = False
        self.current_workspace = None

        print(f"Query mode deactivated for workspace '{workspace}' - returning to command loop")

    def is_active(self):
        """Check if query mode is active"""
        return self.active

    def _verify_llm_connection(self):
        """Verify connection to LLM"""
        debug_print(self.config, "Verifying LLM connection")

        # Check if we're using Ollama
        if hasattr(self.llm_connector, 'check_connection'):
            return self.llm_connector.check_connection()

        # For other connectors, just return True
        return True

    def _generate_response(self, query, docs):
        """
        Generate a response using an LLM with context

        Args:
            query (str): The original query
            docs (list): Relevant documents

        Returns:
            str: The generated response
        """
        debug_print(self.config, "Generating response with LLM")

        # Get LLM model from config
        model = self.config.get('llm.default_model')

        # Use the LLM connector to generate a response
        try:
            return self.llm_connector.generate_with_context(query, docs, model)
        except Exception as e:
            debug_print(self.config, f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _generate_response_no_context(self, query):
        """
        Generate a response using an LLM without context

        Args:
            query (str): The original query

        Returns:
            str: The generated response
        """
        debug_print(self.config, "Generating response with LLM (no context)")

        # Get LLM model from config
        model = self.config.get('llm.default_model')

        # Create a prompt
        prompt = f"You are a document analysis assistant. The user is asking about documents in their collection, but no relevant documents were found. Please provide a helpful response.\n\nUser query: {query}\n\nResponse:"

        # Use the LLM connector to generate a response
        try:
            return self.llm_connector.generate(prompt, model)
        except Exception as e:
            debug_print(self.config, f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def enter_interactive_mode(self):
        """
        Enter an isolated interactive query mode

        This method creates a separate loop for processing queries
        """
        print("Entering interactive query mode. Type /exit to return to main command mode.")

        while True:
            # Use custom prompt showing workspace and LLM model
            llm_model = self.config.get('llm.default_model')
            prompt = f"[{self.current_workspace}][{llm_model}> "

            try:
                # Get user input
                query = input(prompt).strip()

                # Check for exit command
                if query.lower() == '/exit':
                    print("Exiting query mode.")
                    break

                # Skip empty queries
                if not query:
                    continue

                # Process the query
                self.process_query(query)

            except KeyboardInterrupt:
                print("\nReturning to main command mode.")
                break

    def process_query(self, query):
        """
        Process a user query with compact colored output

        Args:
            query (str): The query string
        """
        debug_print(self.config, f"Processing query: {query}")

        if not self.active:
            self.output_manager.print_formatted('feedback', "Query mode is not active", success=False)
            return

        # Handle exit/quit commands to exit query mode
        if query.strip().lower() in ['/exit', '/quit']:
            self.output_manager.print_formatted('feedback', "Exiting query mode")
            self.deactivate()
            return

        # Display the query with compact formatting
        self.output_manager.print_formatted('header', "QUERY")
        print(f"{query}\n")

        # Preprocess query
        processed_query = self.text_processor.preprocess(query)

        # Get k value from config
        k_value = self.config.get('query.k_value')

        # Retrieve documents
        docs = self.storage_manager.query_documents(self.current_workspace, processed_query['processed'], k=k_value)

        if not docs:
            self.output_manager.print_formatted('feedback', "No relevant documents found", success=False)
            response = self._generate_response_no_context(query)

            # Display the response
            self.output_manager.print_formatted('header', "RESPONSE")
            print(f"{response}\n")

            self.output_manager.add_to_buffer(query, response, [])
            return

        # Display retrieved documents header with count
        self.output_manager.print_formatted('subheader', f"RETRIEVED DOCUMENTS ({len(docs)})")

        # Import color codes
        from core.engine.utils import Colors

        # Display document summaries in a compact format
        for i, doc in enumerate(docs):
            source = doc.get('metadata', {}).get('source', 'unknown')
            score = doc.get('relevance_score', 'N/A')

            # Format document with colors
            print(f"\n{Colors.YELLOW}Document {i + 1}: {source}{Colors.RESET}")
            print(f"  {Colors.BRIGHT_WHITE}Relevance:{Colors.RESET} {score:.4f}" if isinstance(score,
                                                                                               float) else f"  {Colors.BRIGHT_WHITE}Relevance:{Colors.RESET} {score}")

            # Show content preview with gray text
            content = doc.get('content', '')
            preview = (content[:200] + '...') if len(content) > 200 else content
            preview_lines = preview.split('\n')

            if preview_lines:
                print(f"  {Colors.BRIGHT_WHITE}Preview:{Colors.RESET}")
                for line in preview_lines[:3]:  # Limit to first 3 lines for compactness
                    print(f"  {Colors.GRAY}{line}{Colors.RESET}")
                if len(preview_lines) > 3:
                    print(f"  {Colors.GRAY}...{Colors.RESET}")

        # Generate response using LLM
        response = self._generate_response(query, docs)

        # Display the response
        self.output_manager.print_formatted('header', "RESPONSE")
        print(f"{response}\n")

        # Save to buffer for potential later saving
        self.output_manager.add_to_buffer(query, response, docs)

