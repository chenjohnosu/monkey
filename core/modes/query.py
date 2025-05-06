"""
Query mode module with LlamaIndex and Ollama integration
Fixed implementation to address crashes in /run query command
"""

import os
import textwrap
import traceback

from core.engine.logging import debug, info, warning, error, trace
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.connectors.connector_factory import ConnectorFactory
from core.engine.utils import ensure_dir
from core.engine.common import safe_execute


class QueryEngine:
    """Interactive query engine using LlamaIndex and Ollama"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the query engine with improved error handling"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)

        # Safely initialize the LLM connector
        try:
            self.factory = ConnectorFactory(config)
            self.llm_connector = self.factory.get_llm_connector()
            if not self.llm_connector:
                warning("Failed to initialize LLM connector")
        except Exception as e:
            error(f"Error initializing LLM connector: {str(e)}")
            import traceback
            error(traceback.format_exc())
            self.factory = None
            self.llm_connector = None
            warning("LLM connector initialization failed - query functionality may be limited")

        self.batch_mode = config.get('system.batch_mode', False)
        self.active = False
        self.current_workspace = None
        self.session_logger = False
        debug(config, "Query engine initialized")

    def activate(self, workspace):
        """Activate query mode for a specific workspace"""
        debug(self.config, f"Activating query mode for workspace '{workspace}'")

        # Check if workspace exists
        data_dir = os.path.join("data", workspace)
        if not os.path.exists(data_dir):
            print(f"Workspace '{workspace}' does not exist or has no vector store")
            return False

        # Set active state
        self.active = True
        self.current_workspace = workspace
        self.session_logger = False  # Reset session logger state

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
        """Deactivate query mode with improved error handling"""
        debug(self.config, "Deactivating query mode")

        # Store the workspace name for the message
        workspace = self.current_workspace

        # Make sure to stop session logging first if it's active
        try:
            if hasattr(self, 'session_logger') and self.session_logger:
                self._stop_session_logging()
        except Exception as e:
            debug(self.config, f"Error stopping session logging: {str(e)}")
            # Continue even if there's an error

        # Reset state
        self.active = False

        # Keep current workspace to allow reactivation with the same workspace
        print(f"Query mode deactivated for workspace '{workspace}' - returning to command loop")

    def is_active(self):
        """Check if query engine is active"""
        return self.active

    def _verify_llm_connection(self):
        """
        Verify LLM connection with improved error handling

        Returns:
            bool: True if connection is verified, False otherwise
        """
        debug(self.config, "Verifying LLM connection")

        # First check if llm_connector exists
        if not hasattr(self, 'llm_connector') or self.llm_connector is None:
            warning("LLM connector not initialized")
            return False

        # Check if connector supports connection verification
        if hasattr(self.llm_connector, 'check_connection'):
            try:
                return self.llm_connector.check_connection()
            except Exception as e:
                error(f"Error checking LLM connection: {str(e)}")
                return False

        # If check_connection is not available, try a minimal test generation
        try:
            model = self.config.get('llm.default_model')
            test_result = self.llm_connector.generate("Test", model=model, max_tokens=5)
            return test_result is not None and len(test_result) > 0
        except Exception as e:
            error(f"Error testing LLM connection: {str(e)}")
            return False

    def _generate_response(self, query, docs):
        """
        Generate response with LLM based on query and retrieved documents

        Args:
            query (str): User query
            docs (List[Dict]): Retrieved documents

        Returns:
            str: Generated response
        """
        debug(self.config, "Generating response with LLM")

        try:
            model = self.config.get('llm.default_model')

            # Check if LLM connector is available
            if not hasattr(self, 'llm_connector') or self.llm_connector is None:
                error("LLM connector not available")
                return "Error: LLM connector not available. Please check your configuration."

            # Try to use generate_with_context if available
            if hasattr(self.llm_connector, 'generate_with_context'):
                return self.llm_connector.generate_with_context(query, docs, model)
            else:
                # Fallback to manual context building
                prompt = f"Query: {query}\n\nContext documents:\n"
                for i, doc in enumerate(docs):
                    content = doc.get('content', '').strip()
                    source = doc.get('metadata', {}).get('source', 'unknown')
                    prompt += f"\n--- Document {i + 1}: {source} ---\n{content[:1000]}\n"

                prompt += f"\nBased on the above documents, please answer the query: {query}"

                return self.llm_connector.generate(prompt, model)
        except Exception as e:
            error(f"Error generating response: {str(e)}")
            import traceback
            error(traceback.format_exc())
            return f"Error generating response: {str(e)}"

    def _generate_response_no_context(self, query):
        """
        Generate response with LLM when no relevant documents are found

        Args:
            query (str): User query

        Returns:
            str: Generated response
        """
        debug(self.config, "Generating response with LLM (no context)")

        try:
            model = self.config.get('llm.default_model')

            # Check if LLM connector is available
            if not hasattr(self, 'llm_connector') or self.llm_connector is None:
                error("LLM connector not available")
                return "Error: LLM connector not available. Please check your configuration."

            # Create a prompt for no-context scenario
            prompt = f"""You are a document analysis assistant. The user is asking about documents in their collection, 
but no relevant documents were found. Please provide a helpful response.

User query: {query}

Response:"""

            return self.llm_connector.generate(prompt, model)
        except Exception as e:
            error(f"Error generating response: {str(e)}")
            import traceback
            error(traceback.format_exc())
            return f"Error generating response: {str(e)}"

    def _log_to_session(self, content):
        """Helper method to log content to active session"""
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            try:
                self.output_manager._write_to_session(content)
            except Exception as e:
                debug(self.config, f"Error writing to session log: {str(e)}")

    def process_query(self, query):
        """Process a user query in the current workspace"""
        debug(self.config, f"Processing query: {query}")

        if not self.active:
            print("Query mode is not active")
            return

        # Log query to system message log
        info(f"QUERY: {query}")

        # Explicitly log the query text if we're saving a session
        if self.session_logger and hasattr(self.output_manager, '_write_to_session'):
            self._log_to_session(f"User Query: {query}")

        try:
            # Preprocess query
            processed_query = self.text_processor.preprocess(query)
            debug(f"Processed query: {processed_query['processed']}")
            debug(f"Detected language: {processed_query['language']}")

            # Get k value from config
            k_value = self.config.get('query.k_value', 5)  # Default to 5 if not set
            debug(f"Using k value: {k_value}")

            # Retrieve documents
            docs = self.storage_manager.query_documents(self.current_workspace, processed_query['processed'], k=k_value)

            if not docs:
                warning("No relevant documents found for query")
                response = self._generate_response_no_context(query)

                # Only print response header if not in batch mode
                if not self.batch_mode:
                    print("\nRESPONSE:")

                import textwrap
                wrapped_response_lines = textwrap.wrap(response, width=80)
                print('\n'.join(wrapped_response_lines) + '\n')

                # Explicitly log the response text to session if active
                self._log_to_session(f"Response (no documents found): {response}")

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
                doc_log += f"Document {i + 1}: {source}, Relevance: {score_str}\n"

                # Show content preview in debug log
                content = doc.get('content', '')
                preview = (content[:200] + '...') if len(content) > 200 else content
                preview_flat = preview.replace('\n', ' ')
                debug(f"  Preview: {preview_flat}")

                # Add preview to session log if active
                doc_log += f"  Preview: {preview_flat}\n"

            # Log the document summaries to session if active
            self._log_to_session(doc_log)

            # Generate response using LLM
            info("Generating response with retrieved documents...")
            response = self._generate_response(query, docs)

            # Only print response header if not in batch mode
            if not self.batch_mode:
                print("\nRESPONSE:")

            import textwrap
            wrapped_response_lines = textwrap.wrap(response, width=80)
            print('\n'.join(wrapped_response_lines) + '\n')

            # Explicitly log the response text to session if active
            self._log_to_session(f"Response: {response}")

            # Save to buffer for potential later saving
            self.output_manager.add_to_buffer(query, response, docs)

        except Exception as e:
            error(f"Error processing query: {str(e)}")
            import traceback
            error(traceback.format_exc())
            print(f"\nError processing query: {str(e)}")

            # Add error to buffer
            self.output_manager.add_to_buffer(query, f"Error: {str(e)}", [])

    def process_one_time_query(self, query_text):
        """
        Process a one-time query without entering interactive mode

        Args:
            query_text (str): The query text to process

        Returns:
            str: Response to the query
        """
        debug(self.config, f"Processing one-time query: {query_text}")

        # Store current active state to restore it later
        was_active = self.is_active()
        workspace = self.current_workspace

        try:
            # Activate query engine if not already active
            if not self.is_active():
                if not self.activate(workspace):
                    return "Failed to activate query engine for the workspace."

            # Process the query with error handling
            try:
                self.process_query(query_text)

                # Get the response from the buffer
                if hasattr(self, 'output_manager') and self.output_manager.buffer:
                    response = self.output_manager.buffer.get('response', '')
                else:
                    response = "Query processed, but no response was buffered."
            except Exception as e:
                error(f"Error processing query: {str(e)}")
                import traceback
                error(traceback.format_exc())
                response = f"Error processing query: {str(e)}"

            return response

        finally:
            # Restore previous state - only deactivate if it wasn't active before
            if not was_active:
                # Gracefully deactivate without affecting the original state
                if hasattr(self, 'session_logger') and self.session_logger:
                    self._stop_session_logging()

                # Reset state without full deactivation
                self.active = False

    def enter_interactive_mode(self):
        """Enter interactive query mode with improved error handling"""
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

                # Process the query with error handling
                try:
                    self.process_query(query)
                except Exception as e:
                    error(f"Error processing query: {str(e)}")
                    import traceback
                    error(traceback.format_exc())
                    print(f"\nError processing query: {str(e)}")

            except KeyboardInterrupt:
                print("\nReturning to main command mode.")
                self.deactivate()
                break

    def _start_session_logging(self):
        """Start query session logging with improved error handling"""
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
                print(f"Warning: Could not start session logging: {str(e)}")
                self.session_logger = False
        else:
            debug(self.config, "OutputManager doesn't support session saving")
            self.session_logger = False

    def _stop_session_logging(self):
        """Stop query session logging with improved error handling"""
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