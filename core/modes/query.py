"""
Query mode module with LlamaIndex and Ollama integration
With improved session logging capability, fixed exit behavior,
and optimized for performance, speed, and accuracy
"""
import os
import textwrap
import time
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple
from core.engine.logging import debug, info, warning, error
from core.engine.storage import StorageManager
from core.engine.output import OutputManager
from core.language.processor import TextProcessor
from core.connectors.connector_factory import ConnectorFactory

class QueryEngine:
    """Interactive query engine using LlamaIndex and Ollama"""

    def __init__(self, config, storage_manager=None, output_manager=None, text_processor=None):
        """Initialize the query engine with optimized configuration"""
        self.config = config
        self.storage_manager = storage_manager or StorageManager(config)
        self.output_manager = output_manager or OutputManager(config)
        self.text_processor = text_processor or TextProcessor(config)
        self.factory = ConnectorFactory(config)
        self.llm_connector = self.factory.get_llm_connector()

        # State variables
        self.active = False
        self.current_workspace = None

        # Unified session logging attribute (fixes inconsistency)
        self.session_logging = False

        # Cache for frequently accessed config values
        self._config_cache = {}

        # Performance optimization settings
        self.relevance_threshold = self._get_config_value('query.relevance_threshold', 0.0)
        self.max_retries = self._get_config_value('query.max_retries', 2)
        self.retry_delay = self._get_config_value('query.retry_delay', 1.0)

        # Document processing optimization
        self.batch_size = self._get_config_value('query.batch_size', 10)

        debug(config, "Query engine initialized with optimized settings")

    def activate(self, workspace):
        """
        Activate query mode for a workspace

        Args:
            workspace (str): The workspace to query
        """
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
        """Deactivate query mode"""
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
        """Check if query mode is active"""
        return self.active

    def _get_config_value(self, key: str, default: Any) -> Any:
        """
        Get a configuration value with caching for performance

        Args:
            key: The configuration key
            default: Default value if key is not found

        Returns:
            The configuration value or default
        """
        # Check cache first
        if key in self._config_cache:
            return self._config_cache[key]

        # Get from config and cache it
        value = self.config.get(key, default)
        self._config_cache[key] = value
        return value

    def _verify_llm_connection(self):
        """Verify connection to LLM"""
        debug(self.config, "Verifying LLM connection")

        # Check if connector supports connection verification
        if hasattr(self.llm_connector, 'check_connection'):
            return self.llm_connector.check_connection()

        # For other connectors, assume connection is valid
        return True

    def _generate_response(self, query: str, docs: List[Dict[str, Any]]) -> str:
        """
        Generate a response using an LLM with context, with retry logic for reliability

        Args:
            query: The original query
            docs: Relevant documents

        Returns:
            The generated response
        """
        debug(self.config, "Generating response with LLM")
        model = self._get_config_value('llm.default_model', 'mistral')

        # Implement retry logic for improved reliability
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                if retries > 0:
                    debug(self.config, f"Retry attempt {retries}/{self.max_retries} for LLM response generation")

                return self.llm_connector.generate_with_context(query, docs, model)

            except Exception as e:
                last_error = e
                debug(self.config, f"Error generating response (attempt {retries+1}/{self.max_retries+1}): {str(e)}")

                if retries < self.max_retries:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2 ** retries)
                    debug(self.config, f"Waiting {wait_time:.2f}s before retry...")
                    time.sleep(wait_time)

                retries += 1

        # If we get here, all retries failed
        error_msg = f"Error generating response after {self.max_retries+1} attempts: {str(last_error)}"
        warning(error_msg)
        return f"I apologize, but I encountered a technical issue while generating a response. Please try again or rephrase your query."

    def _generate_response_no_context(self, query: str) -> str:
        """
        Generate a response using an LLM without context, with retry logic for reliability

        Args:
            query: The original query

        Returns:
            The generated response
        """
        debug(self.config, "Generating response with LLM (no context)")
        model = self._get_config_value('llm.default_model', 'mistral')

        # Create a prompt for no-context scenario
        prompt = f"You are a document analysis assistant. The user is asking about documents in their collection, but no relevant documents were found. Please provide a helpful response.\n\nUser query: {query}\n\nResponse:"

        # Implement retry logic for improved reliability
        retries = 0
        last_error = None

        while retries <= self.max_retries:
            try:
                if retries > 0:
                    debug(self.config, f"Retry attempt {retries}/{self.max_retries} for LLM response generation (no context)")

                return self.llm_connector.generate(prompt, model)

            except Exception as e:
                last_error = e
                debug(self.config, f"Error generating no-context response (attempt {retries+1}/{self.max_retries+1}): {str(e)}")

                if retries < self.max_retries:
                    # Wait before retrying with exponential backoff
                    wait_time = self.retry_delay * (2 ** retries)
                    debug(self.config, f"Waiting {wait_time:.2f}s before retry...")
                    time.sleep(wait_time)

                retries += 1

        # If we get here, all retries failed
        error_msg = f"Error generating no-context response after {self.max_retries+1} attempts: {str(last_error)}"
        warning(error_msg)
        return f"I apologize, but I encountered a technical issue while generating a response. Please try again or rephrase your query."

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
        debug(self.config, "Starting query session logging")

        # Check if output_manager has session functionality
        if hasattr(self.output_manager, 'start_session_saving'):
            try:
                # Start session saving with workspace prefix
                self.output_manager.start_session_saving(self.current_workspace)
                print("Query session logging started. All queries and responses will be saved.")
                self.session_logging = True
            except Exception as e:
                debug(self.config, f"Error starting query session logging: {str(e)}")
                self.session_logging = False
        else:
            debug(self.config, "OutputManager doesn't support session saving")
            self.session_logging = False

    def _stop_session_logging(self):
        """Stop logging query session"""
        debug(self.config, "Stopping query session logging")

        # Check if we're logging and output_manager has the functionality
        if self.session_logging and hasattr(self.output_manager, 'stop_session_saving'):
            try:
                filepath = self.output_manager.stop_session_saving()
                print(f"Query session logging stopped. Session log saved to: {filepath}")
                self.session_logging = False
            except Exception as e:
                print(self.config, f"Error stopping query session logging: {str(e)}")
                print(f"Warning: Could not properly save session log: {str(e)}")
                self.session_logging = False

    def _log_to_session(self, content):
        """Helper method to log content to active session"""
        if self.session_logging and hasattr(self.output_manager, '_write_to_session'):
            self.output_manager._write_to_session(content)

    def process_query(self, query):
        """
        Process a user query with output to main screen only for response
        and logging everything else to system message log
        """
        from core.engine.logging import info, debug, warning, error

        debug(self.config, f"Processing query: {query}")

        if not self.active:
            self.output_manager.print_formatted('feedback', "Query mode is not active", success=False)
            return

        # Log query to system message log
        info(f"QUERY: {query}")

        # Log the query text to session if active
        self._log_to_session(f"User Query: {query}")

        # Preprocess query
        processed_query = self.text_processor.preprocess(query)
        debug(f"Processed query: {processed_query['processed']}")
        debug(f"Detected language: {processed_query['language']}")

        # Get k value from config
        k_value = self.config.get('query.k_value')
        debug(f"Using k value: {k_value}")

        # Retrieve documents with optimized processing
        docs = self.storage_manager.query_documents(self.current_workspace, processed_query['processed'], k=k_value)

        # Filter documents by relevance score if threshold is set
        if self.relevance_threshold > 0 and docs:
            filtered_docs = []
            for doc in docs:
                score = doc.get('relevance_score', 0)
                if isinstance(score, float) and score >= self.relevance_threshold:
                    filtered_docs.append(doc)

            # Log filtering results
            if len(filtered_docs) < len(docs):
                debug(f"Filtered out {len(docs) - len(filtered_docs)} documents below relevance threshold {self.relevance_threshold}")

            docs = filtered_docs

        if not docs:
            warning("No relevant documents found for query")
            response = self._generate_response_no_context(query)

            # Display the response (this is the only output to main screen)
            info("RESPONSE (no context):")
            # Use output_manager to format and print to main screen
            self.output_manager.print_formatted('header', "RESPONSE")
            wrapped_response_lines = textwrap.wrap(response, width=80)
            print('\n'.join(wrapped_response_lines) + '\n')

            # Log the response text to session if active
            self._log_to_session(f"Response (no documents found): {response}")

            self.output_manager.add_to_buffer(query, response, [])
            return

        # Log retrieved documents info to system message log
        info(f"RETRIEVED DOCUMENTS: {len(docs)}")

        # Prepare document log with optimized batch processing
        doc_log = f"Retrieved {len(docs)} documents\n"

        # Process documents in batches for better performance
        batch_size = min(self.batch_size, len(docs))

        # Pre-allocate lists for better memory efficiency
        sources = []
        scores = []
        previews = []

        # Extract document metadata in a single pass
        for i, doc in enumerate(docs):
            # Extract metadata efficiently
            source = doc.get('metadata', {}).get('source', 'unknown')
            score = doc.get('relevance_score', 'N/A')
            score_str = f"{score:.4f}" if isinstance(score, float) else str(score)

            # Generate preview
            content = doc.get('content', '')
            preview = (content[:200] + '...') if len(content) > 200 else content
            preview_flat = preview.replace('\n', ' ')

            # Store for batch logging
            sources.append(source)
            scores.append(score_str)
            previews.append(preview_flat)

            # Log document details
            debug(f"Document {i + 1}: {source} (Relevance: {score_str})")
            debug(f"  Preview: {preview_flat}")

            # Build log entry
            doc_log += f"Document {i + 1}: {source}, Relevance: {score_str}\n"
            doc_log += f"  Preview: {preview_flat}\n"

        # Log the document summaries to session if active
        self._log_to_session(doc_log)

        # Generate response using LLM
        info("Generating response with retrieved documents...")
        response = self._generate_response(query, docs)

        # Display the response (this is the only output to main screen)
        info("RESPONSE:")
        # Use output_manager to format and print to main screen
        self.output_manager.print_formatted('header', "RESPONSE")
        wrapped_response_lines = textwrap.wrap(response, width=80)
        print('\n'.join(wrapped_response_lines) + '\n')

        # Log the response text to session if active
        self._log_to_session(f"Response: {response}")

        # Save to buffer for potential later saving
        self.output_manager.add_to_buffer(query, response, docs)
