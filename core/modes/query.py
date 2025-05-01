"""
Query mode module with LlamaIndex and Ollama integration
"""

import os
import textwrap

from core.engine.logging import debug, info, warning, error
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.connectors.connector_factory import ConnectorFactory
from core.engine.utils import ensure_dir
from core.engine.common import safe_execute


class QueryEngine:
    """Interactive query engine using LlamaIndex and Ollama"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)
        self.factory = ConnectorFactory(config)
        self.llm_connector = self.factory.get_llm_connector()
        self.batch_mode = config.get('system.batch_mode', False)

        self.active = False
        self.current_workspace = None
        self.session_logger = False
        debug(config, "Query engine initialized")

    def activate(self, workspace):
        debug(self.config, f"Activating query mode for workspace '{workspace}'")

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
        debug(self.config, "Deactivating query mode")

        # Store the workspace name for the message
        workspace = self.current_workspace

        # Make sure to stop session logging first if it's active
        self._stop_session_logging()

        # Reset state
        self.active = False
        self.current_workspace = None

        print(f"Query mode deactivated for workspace '{workspace}' - returning to command loop")

    def is_active(self):
        return self.active

    def _verify_llm_connection(self):
        debug(self.config, "Verifying LLM connection")

        # Check if connector supports connection verification
        if hasattr(self.llm_connector, 'check_connection'):
            return self.llm_connector.check_connection()

        # For other connectors, assume connection is valid
        return True

    def _generate_response(self, query, docs):
        debug(self.config, "Generating response with LLM")
        model = self.config.get('llm.default_model')

        try:
            return self.llm_connector.generate_with_context(query, docs, model)
        except Exception as e:
            debug(self.config, f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def _generate_response_no_context(self, query):
        debug(self.config, "Generating response with LLM (no context)")
        model = self.config.get('llm.default_model')

        # Create a prompt for no-context scenario
        prompt = f"You are a document analysis assistant. The user is asking about documents in their collection, but no relevant documents were found. Please provide a helpful response.\n\nUser query: {query}\n\nResponse:"

        try:
            return self.llm_connector.generate(prompt, model)
        except Exception as e:
            debug(self.config, f"Error generating response: {str(e)}")
            return f"Error generating response: {str(e)}"

    def enter_interactive_mode(self):
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
        debug(self.config, "Starting query session logging")

        # Check if output_manager has session functionality
        if hasattr(self.output_manager, 'start_session_saving'):
            try:
                # Start session saving with workspace prefix
                self.output_manager.start_session_saving(self.current_workspace)
                print("Query session logging started. All queries and responses will be saved.")
                self.session_logger = True
            except Exception as e:
                debug(self.config, f"Error starting query session logging: {str(e)}")
                self.session_logger = False
        else:
            debug(self.config, "OutputManager doesn't support session saving")
            self.session_logger = False

    def _stop_session_logging(self):
        debug(self.config, "Stopping query session logging")

        # Check if we're logging and output_manager has the functionality
        if self.session_logger and hasattr(self.output_manager, 'stop_session_saving'):
            try:
                filepath = self.output_manager.stop_session_saving()
                print(f"Query session logging stopped. Session log saved to: {filepath}")
                self.session_logger = False
            except Exception as e:
                debug(self.config, f"Error stopping query session logging: {str(e)}")
                print(f"Warning: Could not properly save session log: {str(e)}")
                self.session_logger = False

    def _log_to_session(self, content):
        """Helper method to log content to active session"""
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            self.output_manager._write_to_session(content)

    def process_query(self, query):
        from core.engine.logging import info, debug, warning, error
        debug(self.config, f"Processing query: {query}")

        if not self.active:
            print("Query mode is not active")
            return

        # Log query to system message log
        info(f"QUERY: {query}")

        # Explicitly log the query text if we're saving a session
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            self.output_manager._write_to_session(f"User Query: {query}")

        # Preprocess query
        processed_query = self.text_processor.preprocess(query)
        debug(f"Processed query: {processed_query['processed']}")
        debug(f"Detected language: {processed_query['language']}")

        # Get k value from config
        k_value = self.config.get('query.k_value')
        debug(f"Using k value: {k_value}")

        # Retrieve documents
        docs = self.storage_manager.query_documents(self.current_workspace, processed_query['processed'], k=k_value)

        if not docs:
            warning("No relevant documents found for query")
            response = self._generate_response_no_context(query)

            # Only print response header if not in batch mode
            if not self.batch_mode:
                print("\nRESPONSE:")

            wrapped_response_lines = textwrap.wrap(response, width=80)
            print('\n'.join(wrapped_response_lines) + '\n')

            # Explicitly log the response text to session if active
            if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
                self.output_manager._write_to_session(f"Response (no documents found): {response}")

            self.output_manager.add_to_buffer(query, response, [])
            return

        # Log retrieved documents info to system message log
        info(f"RETRIEVED DOCUMENTS: {len(docs)}")

        # Detailed document info goes to system log
        doc_log = f"Retrieved {len(docs)} documents\n"
        for i, doc in enumerate(docs):
            source = doc.get('metadata', {}).get('source', 'unknown')
            score = doc.get('relevance_score', 'N/A')

            # Format as readable score
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)

            # Log document details
            debug(f"Document {i + 1}: {source} (Relevance: {score_str})")

            # Add to session log if active
            if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
                doc_log += f"Document {i + 1}: {source}, Relevance: {score_str}\n"

            # Show content preview in debug log
            content = doc.get('content', '')
            preview = (content[:200] + '...') if len(content) > 200 else content
            preview_flat = preview.replace('\n', ' ')
            debug(f"  Preview: {preview_flat}")

            # Add preview to session log if active
            if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
                doc_log += f"  Preview: {preview_flat}\n"

        # Log the document summaries to session if active
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            self.output_manager._write_to_session(doc_log)

        # Generate response using LLM
        info("Generating response with retrieved documents...")
        response = self._generate_response(query, docs)

        # Only print response header if not in batch mode
        if not self.batch_mode:
            print("\nRESPONSE:")

        wrapped_response_lines = textwrap.wrap(response, width=80)
        print('\n'.join(wrapped_response_lines) + '\n')

        # Explicitly log the response text to session if active
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            self.output_manager._write_to_session(f"Response: {response}")

        # Save to buffer for potential later saving
        self.output_manager.add_to_buffer(query, response, docs)

    def process_one_time_query(self, query_text):
        debug(self.config, f"Processing one-time query: {query_text}")

        if not self.active:
            if not self.activate(self.current_workspace):
                return "Failed to activate query mode for the workspace."

        # Process the query
        self.process_query(query_text)

        # Get the response from the buffer
        if hasattr(self, 'output_manager') and self.output_manager.buffer:
            response = self.output_manager.buffer.get('response', '')
        else:
            response = "Query processed, but no response was buffered."

        # Deactivate query mode
        self.deactivate()

        return response