"""
Query mode module with LlamaIndex and Ollama integration
With improved session logging capability and fixed exit behavior
"""

import os
import textwrap

from core.engine.logging import debug_print, info, warning, error
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.connectors.connector_factory import ConnectorFactory
from core.engine.utils import ensure_dir, Colors
from core.engine.common import safe_execute


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
        self.session_logger = None
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

        # Connect to LLM and verify model availability
        if not self._verify_llm_connection():
            print("Warning: LLM connection could not be verified. Will attempt to use when needed.")

        return True

    def deactivate(self):
        """Deactivate query mode"""
        debug_print(self.config, "Deactivating query mode")

        # Store the workspace name for the message
        workspace = self.current_workspace

        # Make sure to stop session logging first if it's active
        self._stop_session_logging()

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

        # Check if connector supports connection verification
        if hasattr(self.llm_connector, 'check_connection'):
            return self.llm_connector.check_connection()

        # For other connectors, assume connection is valid
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
        model = self.config.get('llm.default_model')

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
        model = self.config.get('llm.default_model')

        # Create a prompt for no-context scenario
        prompt = f"You are a document analysis assistant. The user is asking about documents in their collection, but no relevant documents were found. Please provide a helpful response.\n\nUser query: {query}\n\nResponse:"

        try:
            return self.llm_connector.generate(prompt, model)
        except Exception as e:
            debug_print(self.config, f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def enter_interactive_mode(self):
        """
        Enter an isolated interactive query mode
        """
        print("Entering interactive query mode. Type /exit to return to main command mode.")

        # Start session logging if configured
        self._start_session_logging()

        while True:
            # Use custom prompt showing workspace, LLM model and query indicator
            llm_model = self.config.get('llm.default_model')
            prompt = f"[{self.current_workspace}][{llm_model}][query]> "

            try:
                # Get user input
                query = input(prompt).strip()

                # Check for exit command
                if query.lower() in ['/exit', '/quit']:
                    print("Exiting query mode.")
                    self.deactivate()
                    break

                # Skip empty queries
                if not query:
                    continue

                # Process the query
                self.process_query(query)

            except KeyboardInterrupt:
                print("\nReturning to main command mode.")
                self.deactivate()
                break

    def _start_session_logging(self):
        """Start logging query session to a file"""
        debug_print(self.config, "Starting query session logging")

        # Check if output_manager has session functionality
        if hasattr(self.output_manager, 'start_session_saving'):
            try:
                # Start session saving with workspace prefix
                self.output_manager.start_session_saving(self.current_workspace)
                print("Query session logging started. All queries and responses will be saved.")
                self.session_logger = True
            except Exception as e:
                debug_print(self.config, f"Error starting query session logging: {str(e)}")
                self.session_logger = False
        else:
            debug_print(self.config, "OutputManager doesn't support session saving")
            self.session_logger = False

    def _stop_session_logging(self):
        """Stop logging query session"""
        debug_print(self.config, "Stopping query session logging")

        # Check if we're logging and output_manager has the functionality
        if self.session_logger and hasattr(self.output_manager, 'stop_session_saving'):
            try:
                filepath = self.output_manager.stop_session_saving()
                print(f"Query session logging stopped. Session log saved to: {filepath}")
                self.session_logger = False
            except Exception as e:
                debug_print(self.config, f"Error stopping query session logging: {str(e)}")
                print(f"Warning: Could not properly save session log: {str(e)}")
                self.session_logger = False

    def _log_to_session(self, content):
        """Helper method to log content to active session"""
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            self.output_manager._write_to_session(content)

    def process_query(self, query):
        """
        Process a user query with compact colored output and properly log all content

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

        # Log the query text
        self._log_to_session(f"User Query: {query}")

        # Preprocess query
        processed_query = self.text_processor.preprocess(query)

        # Get k value from config
        k_value = self.config.get('query.k_value')

        # Retrieve documents
        docs = self.storage_manager.query_documents(self.current_workspace, processed_query['processed'], k=k_value)

        # If no relevant documents found, generate response without context
        if not docs:
            self.output_manager.print_formatted('feedback', "No relevant documents found", success=False)
            response = self._generate_response_no_context(query)

            # Display the response
            self.output_manager.print_formatted('header', "RESPONSE")
            print(f"{response}\n")

            # Log the response text
            self._log_to_session(f"Response (no documents found): {response}")

            # Add to buffer for potential later saving
            self.output_manager.add_to_buffer(query, response, [])
            return

        # Display retrieved documents header with count
        self.output_manager.print_formatted('subheader', f"RETRIEVED DOCUMENTS ({len(docs)})")

        # Import color codes
        from core.engine.utils import Colors

        # Create document summary log
        doc_log = f"Retrieved {len(docs)} documents:\n"

        # Display document summaries in a compact format
        for i, doc in enumerate(docs):
            source = doc.get('metadata', {}).get('source', 'unknown')
            score = doc.get('relevance_score', 'N/A')

            # Format document with colors
            print(f"\n{Colors.YELLOW}Document {i + 1}: {source}{Colors.RESET}")
            print(f"  {Colors.BRIGHT_WHITE}Relevance:{Colors.RESET} {score:.4f}" if isinstance(score,
                                                                                               float) else f"  {Colors.BRIGHT_WHITE}Relevance:{Colors.RESET} {score}")

            # Add to log
            doc_log += f"Document {i + 1}: {source}, Relevance: {score if isinstance(score, str) else f'{score:.4f}'}\n"

            # Show content preview with gray text
            content = doc.get('content', '')
            preview = (content[:200] + '...') if len(content) > 200 else content
            preview_lines = preview.split('\n')

            if preview_lines:
                print(f"  {Colors.BRIGHT_WHITE}Preview:{Colors.RESET}")
                doc_log += "  Preview: {}\n".format(preview.replace('\n', ' '))

                for line in preview_lines[:3]:  # Limit to first 3 lines for compactness
                    print(f"  {Colors.GRAY}{line}{Colors.RESET}")
                if len(preview_lines) > 3:
                    print(f"  {Colors.GRAY}...{Colors.RESET}")

        # Log the document summaries
        self._log_to_session(doc_log)

        # Generate response using LLM
        response = self._generate_response(query, docs)

        # Display the response
        self.output_manager.print_formatted('header', "RESPONSE")
        print(f"{response}\n")

        # Log the response text
        self._log_to_session(f"Response: {response}")

        # Save to buffer for potential later saving
        self.output_manager.add_to_buffer(query, response, docs)