"""
Command line interface for the document analysis toolkit
"""

import os
import shlex
from core.engine.logging import debug, error, warning, info, trace, get_logger, debug_print
from core.engine.storage import StorageManager, VectorStoreInspector
from core.engine.output import OutputManager
import datetime
from core.engine.interpreter import AnalysisInterpreter




class CommandProcessor:

    # CLI
    def __init__(self, config):
        """Initialize CommandProcessor with configuration"""
        self.config = config
        self.running = True
        self.current_workspace = config.get('workspace.default')

        # Create shared instances
        self.storage_manager = StorageManager(config)
        self.output_manager = OutputManager(config)

        # Language processing components
        from core.language.processor import TextProcessor
        self.text_processor = TextProcessor(config)

        # Initialize components with shared instances
        from core.modes.themes import ThemeAnalyzer
        from core.modes.query import QueryEngine
        from core.modes.grind import FileProcessor
        from core.modes.merge import VectorStoreMerger
        from core.modes.sentiment import SentimentAnalyzer
        from core.modes.topic import TopicModeler

        self.theme_analyzer = ThemeAnalyzer(config, self.storage_manager, self.output_manager, self.text_processor)
        self.query_engine = QueryEngine(config, self.storage_manager, self.output_manager, self.text_processor)
        self.file_processor = FileProcessor(config, self.storage_manager, self.text_processor)
        self.vector_store_merger = VectorStoreMerger(config, self.storage_manager)

        self.sentiment_analyzer = SentimentAnalyzer(config, self.storage_manager, self.output_manager,
                                                    self.text_processor)
        self.topic_modeler = TopicModeler(config, self.storage_manager, self.output_manager, self.text_processor)

        # Initialize the vector store inspector
        self.vector_store_inspector = VectorStoreInspector(config, self.storage_manager)

        self.saving_session = False

        self.loaded_workspaces = [self.current_workspace]  # Track loaded workspaces
        self.active_guide = None  # Track the active guide

        info("CommandProcessor initialized")
    def start(self):
        """Start the command processing loop"""
        debug("Starting command processing loop")
        info(f"Monkey v{self.config.get_version()} initialized. Type /help for available commands.")

        try:
            while self.running:
                # Show both workspace and current LLM model in the prompt
                llm_model = self.config.get('llm.default_model')

                # Add [query] indicator when in query mode
                if hasattr(self, 'query_engine') and self.query_engine.is_active():
                    prompt = f"[{self.current_workspace}][{llm_model}][query]> "
                else:
                    prompt = f"[{self.current_workspace}][{llm_model}]> "

                user_input = input(prompt)
                if user_input.strip():
                    self.process_command(user_input)
        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            debug_print(self.config, "Command processing loop ended")
    def _handle_system_command(self, command, args):
        """Handle system commands using command dispatcher pattern"""
        debug_print(self.config, f"Handling system command: {command} with args: {args}")

        # Handle aliases
        command = self._resolve_alias(command)

        # Special handling for exit/quit in query mode
        if (command == 'exit' or command == 'quit') and hasattr(self, 'query_engine') and self.query_engine.is_active():
            # Just exit query mode, not the whole application
            debug_print(self.config, "Exit/quit command in query mode - deactivating query mode")
            self.query_engine.deactivate()
            return

        # Command dispatch dictionary
        command_handlers = {
            'quit': self._cmd_quit,
            'exit': self._cmd_quit,
            'help': self._cmd_help,
            'show': self._cmd_show,
            'run': self._cmd_run,
            'load': self._cmd_load,
            'save': self._cmd_save,
            'config': self._cmd_config,
            'inspect': self._cmd_inspect,
            'explain': self._cmd_explain,
            'clear': self._cmd_clear
        }

        # Get the handler for this command
        handler = command_handlers.get(command)

        if handler:
            # Call the handler with the arguments
            handler(args)
        else:
            # Command not found
            self.output_manager.print_formatted('feedback',
                                                f"Unknown command: /{command}",
                                                success=False)
    def _format_size(self, size_bytes):
        """Format file size in a human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024
    def _resolve_alias(self, command):
        """Resolve command aliases to full commands with dictionary lookup"""
        aliases = {
            'q': 'quit',
            'c': 'config',
            'l': 'load',
            'r': 'run',
            's': 'save',
            'h': 'help',
            'i': 'inspect',
            'e': 'explain',
            'cl': 'clear'
        }
        return aliases.get(command, command)
    def process_command(self, command_string):
        """
        Process a command string with improved error handling and command parsing

        Args:
            command_string (str): The command to process
        """
        debug_print(self.config, f"Processing command: {command_string}")

        # Skip empty commands
        if not command_string.strip():
            return

        # Echo the command to the output manager
        self.output_manager.print_formatted('command', command_string)

        try:
            # Parse the command safely
            parsed_command = self._parse_command(command_string)

            if not parsed_command:
                return

            command, args = parsed_command

            # Process the command based on type
            self._process_parsed_command(command, args)

        except Exception as e:
            # Handle any unexpected errors
            self.output_manager.print_formatted('feedback',
                                                f"Error processing command: {str(e)}",
                                                success=False)

            if self.config.get('system.debug'):
                import traceback
                debug_print(self.config, traceback.format_exc())
    def _parse_command(self, command_string):
        """
        Parse a command string into command and arguments

        Args:
            command_string (str): The command to parse

        Returns:
            tuple: (command, args) or None if parsing failed
        """
        try:
            # Use shlex to properly handle quoted arguments
            import shlex
            tokens = shlex.split(command_string)

            if not tokens:
                return None

            command = tokens[0].lower()
            args = tokens[1:]

            return (command, args)

        except Exception as e:
            self.output_manager.print_formatted('feedback',
                                                f"Error parsing command: {str(e)}",
                                                success=False)
            return None
    def _process_parsed_command(self, command, args):
        """
        Process a parsed command

        Args:
            command (str): The command to process
            args (list): Command arguments
        """
        # Handle system commands (starting with /)
        if command.startswith('/'):
            # Always process system commands, even in query mode
            self._handle_system_command(command[1:], args)
        else:
            # Treat as a query in interactive mode
            if hasattr(self, 'query_engine') and self.query_engine.is_active():
                # Reconstruct original command for query processing
                query = command
                if args:
                    query += ' ' + ' '.join(args)
                self.query_engine.process_query(query)
            else:
                self.output_manager.print_formatted('feedback',
                                                    "Not in query mode. Use '/run query' to enter interactive query mode.",
                                                    success=False)

    # Command: Quit
    def _cmd_quit(self, args=None):
        """Handle the quit command with proper argument handling"""
        debug_print(self.config, "Quit command received")
        self.running = False

    # Command: Help
    def _cmd_help(self, args):
        """Display help information with structured help topics"""
        debug_print(self.config, f"Help command with args: {args}")

        if not args:
            # Display main help overview
            print(self.HELP_OVERVIEW)
        elif args[0] in self.HELP_TOPICS:
            # Display specific help topic
            print(self.HELP_TOPICS[args[0]])
        else:
            print(f"No specific help available for '{args[0]}'")
            print("Available help topics: " + ", ".join(self.HELP_TOPICS.keys()))

    # Command: Run
    def _cmd_run(self, args):
        """Handle the run command using a command pattern approach"""
        debug_print(self.config, f"Run command with args: {args}")

        if not args:
            self.output_manager.print_formatted('feedback',
                                                "Usage: /run [themes|query|grind|update|scan|merge|sentiment|topics] [options]",
                                                success=False)
            return

        subcommand = args[0].lower()

        # Dictionary mapping subcommands to handler methods
        run_handlers = {
            'themes': self._run_themes,
            'query': self._run_query,
            'grind': self._run_grind,
            'update': self._run_update,
            'scan': self._run_scan,
            'merge': self._run_merge,
            'sentiment': self._run_sentiment,
            'topics': self._run_topics
        }

        # Get the handler for this subcommand
        handler = run_handlers.get(subcommand)

        if handler:
            # Pass remaining arguments to the handler
            handler(args[1:])
        else:
            self.output_manager.print_formatted('feedback',
                                                f"Unknown run subcommand: {subcommand}",
                                                success=False)
    def _run_themes(self, args):
        """Run theme analysis with specified method"""
        method = args[0] if args else 'all'

        self.output_manager.print_formatted('header', f"Running Theme Analysis: {method.upper()}")
        self.theme_analyzer.analyze(self.current_workspace, method)
    def _run_query(self, args):
        """Enter interactive query mode"""
        self.output_manager.print_formatted('header', "Entering Interactive Query Mode")

        if self.query_engine.activate(self.current_workspace):
            self.query_engine.enter_interactive_mode()
        else:
            self.output_manager.print_formatted('feedback',
                                                "Failed to activate query engine",
                                                success=False)
    def _run_grind(self, args):
        """Process files in workspace to create initial database"""
        self.output_manager.print_formatted('header', f"Processing Files for Workspace: {self.current_workspace}")
        self.file_processor.process_workspace(self.current_workspace)
    def _run_update(self, args):
        """Update workspace with new or modified files"""
        self.output_manager.print_formatted('header', f"Updating Workspace: {self.current_workspace}")
        self.file_processor.update_workspace(self.current_workspace)
    def _run_scan(self, args):
        """Scan workspace for new or updated files"""
        detailed = len(args) > 0 and args[0].lower() == 'detailed'
        mode = "Detailed" if detailed else "Standard"

        self.output_manager.print_formatted('header', f"{mode} Workspace Scan: {self.current_workspace}")
        self.file_processor.scan_workspace(self.current_workspace, detailed)
    def _run_merge(self, args):
        """Merge source workspace into current workspace"""
        if not args:
            self.output_manager.print_formatted('feedback',
                                                "Usage: /run merge <source_workspace>",
                                                success=False)
            return

        source_workspace = args[0]

        self.output_manager.print_formatted('header',
                                            f"Merging Workspace: {source_workspace} → {self.current_workspace}")
        self.vector_store_merger.merge(source_workspace, self.current_workspace)
    def _run_sentiment(self, args):
        """Run sentiment analysis with specified method"""
        method = args[0] if args else 'all'

        self.output_manager.print_formatted('header', f"Running Sentiment Analysis: {method.upper()}")
        self.sentiment_analyzer.analyze(self.current_workspace, method)
    def _run_topics(self, args):
        method = args[0] if args else 'all'
        self.output_manager.print_formatted('header', f"Running Topic Modeling: {method.upper()}")
        self.topic_modeler.analyze(self.current_workspace, method)

    # Command Save
    def _cmd_save(self, args):
        """Handle the save command"""
        debug_print(self.config, f"Save command with args: {args}")

        if not args:
            print("Usage: /save [start|stop|buffer]")
            return

        subcommand = args[0].lower()

        if subcommand == 'start':
            self._start_saving()
        elif subcommand == 'stop':
            self._stop_saving()
        elif subcommand == 'buffer':
            self._save_buffer()
        else:
            print(f"Unknown save subcommand: {subcommand}")
    def _start_saving(self):
        """Start saving session output"""
        debug_print(self.config, "Starting session saving")

        if self.saving_session:
            print("Session saving is already active")
            return

        self.saving_session = True
        self.output_manager.start_session_saving(self.current_workspace)
        print("Session saving started")
    def _stop_saving(self):
        """Stop saving session output"""
        debug_print(self.config, "Stopping session saving")

        if not self.saving_session:
            print("Session saving is not active")
            return

        self.saving_session = False
        filepath = self.output_manager.stop_session_saving()
        print(f"Session saving stopped. Output saved to: {filepath}")
    def _save_buffer(self):
        """Save last command output buffer"""
        debug_print(self.config, "Saving output buffer")

        filepath = self.output_manager.save_buffer(self.current_workspace)
        if filepath:
            print(f"Buffer saved to: {filepath}")
        else:
            print("No buffer to save")

    # Command: Config
    def _cmd_config(self, args):
        """Handle the config command using a command pattern approach"""
        debug_print(self.config, f"Config command with args: {args}")

        if not args:
            self.output_manager.print_formatted('feedback',
                                                "Usage: /config [llm|embed|kval|debug|storage|output|guide] [value]",
                                                success=False)
            return

        subcommand = args[0].lower()

        # Dictionary of config subcommand handlers
        config_handlers = {
            'llm': self._config_llm,
            'embed': self._config_embed,
            'kval': self._config_kval,
            'debug': self._config_debug,
            'storage': self._config_storage,
            'output': self._config_output,
            'guide': self._config_guide
        }

        handler = config_handlers.get(subcommand)

        if handler:
            # Pass remaining arguments to the handler
            handler(args[1:])
        else:
            self.output_manager.print_formatted('feedback',
                                                f"Unknown config subcommand: {subcommand}",
                                                success=False)
    def _config_llm(self, args):
        """Configure LLM model"""
        if not args:
            current_model = self.config.get('llm.default_model')
            self.output_manager.print_formatted('kv', current_model, key="Current LLM model")
            return

        self.config.set('llm.default_model', args[0])
        self.output_manager.print_formatted('feedback',
                                            f"LLM model set to: {args[0]}",
                                            success=True)
    def _config_embed(self, args):
        """Configure embedding model"""
        if not args:
            current_model = self.config.get('embedding.default_model')
            self.output_manager.print_formatted('kv', current_model, key="Current embedding model")
            return

        self.config.set('embedding.default_model', args[0])
        self.output_manager.print_formatted('feedback',
                                            f"Embedding model set to: {args[0]}",
                                            success=True)
    def _config_kval(self, args):
        """Configure k-value for retrieval"""
        if not args:
            current_value = self.config.get('query.k_value')
            self.output_manager.print_formatted('kv', current_value, key="Current k value")
            return

        try:
            k_value = int(args[0])
            self.config.set('query.k_value', k_value)
            self.output_manager.print_formatted('feedback',
                                                f"K value set to: {k_value}",
                                                success=True)
        except ValueError:
            self.output_manager.print_formatted('feedback',
                                                "K value must be an integer",
                                                success=False)
    def _config_debug(self, args):
        """Configure debug mode"""
        if not args:
            debug_status = 'Enabled' if self.config.get('system.debug') else 'Disabled'
            self.output_manager.print_formatted('kv', debug_status, key="Debug mode")
            return

        debug_value = args[0].lower() == 'on'
        self.config.set('system.debug', debug_value)
        debug_status = 'Enabled' if debug_value else 'Disabled'
        self.output_manager.print_formatted('feedback',
                                            f"Debug mode: {debug_status}",
                                            success=True)
    def _config_storage(self, args):
        """Configure storage backend"""
        if not args:
            current_backend = self.config.get('storage.vector_store')
            self.output_manager.print_formatted('kv', current_backend, key="Current storage backend")
            return

        valid_backends = ['llama_index', 'haystack', 'chroma']
        if args[0] not in valid_backends:
            self.output_manager.print_formatted('feedback',
                                                f"Storage backend must be one of: {', '.join(valid_backends)}",
                                                success=False)
            return

        self.config.set('storage.vector_store', args[0])
        self.output_manager.print_formatted('feedback',
                                            f"Storage backend set to: {args[0]}",
                                            success=True)
    def _config_output(self, args):
        """Configure output format"""
        if not args:
            current_format = self.config.get('system.output_format')
            self.output_manager.print_formatted('kv', current_format, key="Current output format")
            return

        valid_formats = ['txt', 'md', 'json']
        if args[0] not in valid_formats:
            self.output_manager.print_formatted('feedback',
                                                f"Output format must be one of: {', '.join(valid_formats)}",
                                                success=False)
            return

        self.config.set('system.output_format', args[0])
        self.output_manager.print_formatted('feedback',
                                            f"Output format set to: {args[0]}",
                                            success=True)
    def _config_guide(self, args):
        """Configure active guide"""
        if not args:
            guide_name = self.active_guide or 'None'
            self.output_manager.print_formatted('kv', guide_name, key="Current active guide")
            return

        self._load_guide(args[0])

    # Command: Inspect
    def _cmd_inspect(self, args):
        """Handle the inspect command"""
        debug_print(self.config, f"Inspect command with args: {args}")

        if not args:
            print("Usage: /inspect [workspace|ws|documents|vectorstore|vdb|query|rebuild|fix]")
            return

        subcommand = args[0].lower()

        # Handle aliases for workspace and vectorstore
        if subcommand == 'ws':
            subcommand = 'workspace'
        elif subcommand == 'vdb':
            subcommand = 'vectorstore'

        if subcommand == 'workspace':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            self.vector_store_inspector.inspect_workspace(workspace)
        elif subcommand == 'documents':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            limit = 5
            if len(args) > 2 and args[2].isdigit():
                limit = int(args[2])
            self.vector_store_inspector.dump_document_content(workspace, limit)
        elif subcommand == 'vectorstore':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            self.vector_store_inspector.dump_vector_store(workspace)
            # NEW: Add LlamaIndex specific inspection
            if self.config.get('storage.vector_store') == 'llama_index':
                try:
                    from core.connectors.llama_index_connector import LlamaIndexConnector
                    llama_connector = LlamaIndexConnector(self.config)
                    llama_connector.inspect_index_store(workspace)
                except Exception as e:
                    error(
                        f"Error inspecting LlamaIndex store: {str(e)}")
        elif subcommand == 'query':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            query = args[2] if len(args) > 2 else "test"
            self.vector_store_inspector.test_query_pipeline(workspace, query)
        elif subcommand == 'rebuild':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            self.vector_store_inspector.rebuild_vector_store(workspace)
        elif subcommand == 'fix':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            (warning(f"Attempting to fix vector store issues for workspace '{workspace}'..."))
            self.vector_store_inspector.fix_common_issues(workspace)
        elif subcommand == 'metadata':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            test_query = args[2] if len(args) > 2 else "test"
            limit = int(args[3]) if len(args) > 3 and args[3].isdigit() else 3
            self._inspect_metadata(workspace, test_query, limit)
        elif subcommand == 'migrate':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            print(f"Migrating vector store for workspace '{workspace}'...")
            success = self.vector_store_inspector.migrate_vector_store(workspace)
            if success:
                print("Migration completed successfully.")
            else:
                print("Migration not needed or failed. Check logs for details.")
        elif subcommand == 'reembed':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            model = args[2] if len(args) > 2 else 'jina-zh'
            print(f"Rebuilding vector store for workspace '{workspace}' using '{model}' embedding model...")
            self.config.set('embedding.default_model', model)
            self.vector_store_inspector.rebuild_vector_store(workspace)
        else:
            error(f"Unknown inspect subcommand: {subcommand}")
    def _inspect_metadata(self, workspace, query="test", limit=3):
        """Inspect raw metadata returned from vector store"""
        print(f"\nInspecting vector store metadata for workspace: {workspace}")
        debug(f"Test query: '{query}'")

        # Load vector store
        loaded = self.storage_manager.load_vector_store(workspace)
        if not loaded:
            error("Failed to load vector store")
            return

        # Perform query without normalization
        try:
            # Direct access to connector to bypass normalization
            raw_results = self.storage_manager.connector.query(workspace, query, limit)

            if not raw_results:
                print("No results found")
                return

            print(f"\nRaw metadata from {len(raw_results)} documents:")
            for i, doc in enumerate(raw_results):
                print(f"\nDocument {i + 1}:")
                if 'metadata' in doc:
                    import pprint
                    pprint.pprint(doc['metadata'])
                else:
                    print("No metadata found")

        except Exception as e:
            error(f"Error: {str(e)}")

    # Command: Explain
    def _cmd_explain(self, args):
        """Handle the explain command for analysis interpretation"""
        debug_print(self.config, f"Explain command with args: {args}")

        if not args:
            print("Usage: /explain [themes|topics|sentiment|session] [question]")
            return

        # The first argument is the subcommand
        subcommand = args[0].lower()

        # Valid subcommands (include both singular and plural forms)
        valid_subcommands = {
            # Map various input forms to standardized forms
            'theme': 'themes',
            'themes': 'themes',
            'topic': 'topics',
            'topics': 'topics',
            'sentiment': 'sentiment',
            'session': 'session'
        }

        # Check if the subcommand is valid and normalize it
        if subcommand not in valid_subcommands:
            error(f"Invalid subcommand: {subcommand}")
            error(f"Must be one of: themes, topics, sentiment, session")
            return

        # Normalize the subcommand to the standard form
        normalized_subcommand = valid_subcommands[subcommand]

        # Get the question (everything after the subcommand)
        question = ' '.join(args[1:]) if len(args) > 1 else None

        # Initialize interpreter if needed
        if not hasattr(self, 'analysis_interpreter'):
            from core.engine.interpreter import AnalysisInterpreter
            self.analysis_interpreter = AnalysisInterpreter(
                self.config,
                self.storage_manager,
                self.output_manager,
                self.text_processor
            )

        # Run the appropriate interpretation based on subcommand
        try:
            workspace = self.current_workspace
            self.output_manager.print_formatted('feedback',
                                                f"Analyzing {normalized_subcommand} results, please wait...")

            interpretation = self.analysis_interpreter.interpret_analysis(
                workspace, normalized_subcommand, query=question
            )

            # Display interpretation with proper formatting
            self.output_manager.print_formatted('header',
                                                f"LLM EXPLANATION OF {normalized_subcommand.upper()} ANALYSIS")
            print(interpretation)

            # Save interpretation to file
            output_format = self.config.get('system.output_format', 'txt')
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"explain_{normalized_subcommand}_{timestamp}.{output_format}"

            logs_dir = os.path.join('logs', self.current_workspace)
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            filepath = os.path.join(logs_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                if output_format == 'json':
                    import json
                    json.dump({
                        'timestamp': datetime.datetime.now().isoformat(),
                        'analysis_type': normalized_subcommand,
                        'question': question,
                        'explanation': interpretation
                    }, f, indent=2)
                else:
                    f.write(f"=== LLM EXPLANATION OF {normalized_subcommand.upper()} ANALYSIS ===\n\n")
                    f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
                    if question:
                        f.write(f"Question: {question}\n")
                    f.write(f"\n{interpretation}\n")

            self.output_manager.print_formatted('feedback', f"Explanation saved to: {filepath}", success=True)

        except Exception as e:
            print(f"Error generating explanation: {str(e)}")
            import traceback
            traceback.print_exc()

    # Command: Show
    def _cmd_show(self, args):
        """Handle the show command"""
        debug_print(self.config, f"Show command with args: {args}")

        if not args:
            print("Usage: /show [status|cuda|config|ws|files|guide]")
            return

        subcommand = args[0].lower()

        if subcommand == 'status':
            self._show_status()
        elif subcommand == 'cuda':
            self._show_cuda()
        elif subcommand == 'config':
            self._show_config()
        elif subcommand == 'ws' or subcommand == 'workspace':
            self._show_workspace()
        elif subcommand == 'files':
            self._show_files()
        elif subcommand == 'guide' or subcommand == 'guides':
            self._show_guides()
        elif subcommand == 'embed':
            if len(args) < 2:
                print(f"Current embedding model: {self.config.get('embedding.default_model')}")
                print("Available models: multilingual-e5, mixbread, bge, jina-zh")
                return
            self.config.set('embedding.default_model', args[1])
            print(f"Embedding model set to: {args[1]}")
        else:
            print(f"Unknown show subcommand: {subcommand}")
    def _show_cuda(self):
        """Show CUDA status"""
        debug_print(self.config, "Showing CUDA status")
        from core.engine.cuda import check_cuda_status
        check_cuda_status(self.config)
    def _show_config(self):
        """Show configuration details"""
        debug_print(self.config, "Showing configuration")
        self.config.display()
    def _show_guides(self):
        """Show available guides"""
        print("Showing available guides")

        try:
            with open("guides.txt", "r", encoding="utf-8") as file:
                content = file.read()

            import re
            # Find all guide tags
            guide_tags = re.findall(r'<([^>]+)>.*?</\1>', content, re.DOTALL)

            if not guide_tags:
                error("No guides found in guides.txt")
                return

            print("Available Guides:")
            for guide in sorted(guide_tags):
                if guide == self.active_guide:
                    print(f"  * {guide} (active)")
                else:
                    print(f"    {guide}")

            info("\nUse '/config guide <name>' to set a guide as active")

        except FileNotFoundError:
            error("guides.txt file not found")
        except Exception as e:
            error(f"Error reading guides: {str(e)}")
    def _show_status(self):
        """Show system status with compact colored formatting"""
        debug_print(self.config, "Showing system status")

        self.output_manager.print_formatted('header', "SYSTEM STATUS")

        self.output_manager.print_formatted('kv', self.current_workspace, key="Current Workspace", indent=2)
        self.output_manager.print_formatted('kv', self.config.get('llm.default_model'), key="LLM Model", indent=2)
        self.output_manager.print_formatted('kv', self.config.get('embedding.default_model'), key="Embedding Model",
                                            indent=2)

        debug_status = 'Enabled' if self.config.get('system.debug') else 'Disabled'
        self.output_manager.print_formatted('kv', debug_status, key="Debug Mode", indent=2)

        self.output_manager.print_formatted('kv', self.config.get('system.output_format'), key="Output Format",
                                            indent=2)
        self.output_manager.print_formatted('kv', self.config.get('hardware.device'), key="Device", indent=2)

        session_status = 'Active' if self.saving_session else 'Inactive'
        self.output_manager.print_formatted('kv', session_status, key="Session Saving", indent=2)
    def _show_workspace(self):
        """Show workspace details with compact colored formatting"""
        debug_print(self.config, "Showing workspace details")

        self.output_manager.print_formatted('header', "WORKSPACES")

        for ws in self.loaded_workspaces:
            if ws == self.current_workspace:
                self.output_manager.print_formatted('list', f"{ws} (active)", indent=2)
            else:
                self.output_manager.print_formatted('list', ws, indent=2)

        # Get additional workspace stats if available
        if hasattr(self, 'storage_manager'):
            stats = self.storage_manager.get_workspace_stats(self.current_workspace)
            if stats and isinstance(stats, dict):
                self.output_manager.print_formatted('subheader', f"Details for {self.current_workspace}")

                # Use dict.get() to safely retrieve values with defaults if keys don't exist
                self.output_manager.print_formatted('kv', stats.get('doc_count', 0), key="Document Count", indent=2)
                self.output_manager.print_formatted('kv', stats.get('embedding_count', 0), key="Embeddings", indent=2)
                self.output_manager.print_formatted('kv', stats.get('last_updated', 'Unknown'), key="Last Updated",
                                                    indent=2)
                self.output_manager.print_formatted('kv', stats.get('embedding_model', 'Unknown'),
                                                    key="Embedding Model", indent=2)

                # Show language distribution
                languages = stats.get('languages', {})
                if languages:
                    self.output_manager.print_formatted('mini_header', "Language Distribution")
                    for lang, count in languages.items():
                        self.output_manager.print_formatted('kv', count, key=lang, indent=4)
    def _show_files(self):
        """Show files in the current workspace with compact colored formatting"""
        debug_print(self.config, "Showing workspace files")

        import os
        import datetime
        doc_dir = os.path.join("body", self.current_workspace)

        if not os.path.exists(doc_dir):
            self.output_manager.print_formatted('feedback', f"Document directory does not exist: {doc_dir}",
                                                success=False)
            return

        self.output_manager.print_formatted('header', f"FILES IN WORKSPACE: {self.current_workspace}")

        files = []
        for root, _, filenames in os.walk(doc_dir):
            for filename in filenames:
                filepath = os.path.join(root, filename)
                rel_path = os.path.relpath(filepath, doc_dir)
                size = os.path.getsize(filepath)
                modified = os.path.getmtime(filepath)
                mod_time = datetime.datetime.fromtimestamp(modified).strftime('%Y-%m-%d %H:%M:%S')

                files.append((rel_path, size, mod_time))

        if not files:
            self.output_manager.print_formatted('feedback', "No files found", success=False)
            return

        # Print file list header
        self.output_manager.print_formatted('kv', "Filename", key="Path", indent=2)
        self.output_manager.print_formatted('kv', "Size", key="Size", indent=2)
        self.output_manager.print_formatted('kv', "Modified", key="Last Modified", indent=2)
        print()  # Add a blank line for readability

        # Print each file with consistent formatting
        for file_path, size, mod_time in sorted(files):
            self.output_manager.print_formatted('list', f"{file_path} ({self._format_size(size)}) - {mod_time}",
                                                indent=2)

    # Command: Load
    def _cmd_load(self, args):
        """Handle the load command"""
        debug_print(self.config, f"Load command with args: {args}")

        if not args:
            print("Usage: /load <workspace>")
            return

        # Simplified load command - always for workspace
        workspace = args[0]
        self._load_workspace(workspace)
    def _load_workspace(self, workspace):
        """
        Load a workspace with confirmation for new workspaces

        Args:
            workspace (str): The workspace to load
        """
        debug_print(self.config, f"Loading workspace: {workspace}")

        # Check if workspace directories exist
        import os
        data_dir = os.path.join("data", workspace)
        body_dir = os.path.join("body", workspace)

        # Check if this is a new workspace
        is_new_workspace = not os.path.exists(data_dir) and not os.path.exists(body_dir)

        if is_new_workspace:
            # Ask for confirmation before creating a new workspace
            confirm = input(f"Workspace '{workspace}' does not exist. Create it? (y/n): ").strip().lower()
            if confirm != 'y' and confirm != 'yes':
                warning(f"Cancelled creation of workspace '{workspace}'")
                return

        # Create necessary directories
        for directory in [data_dir, body_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(f"Created directory: {directory}")

        # Add to loaded workspaces if not already loaded
        if workspace not in self.loaded_workspaces:
            self.loaded_workspaces.append(workspace)
            print(f"Workspace '{workspace}' loaded")
        else:
            print(f"Workspace '{workspace}' is already loaded")

        # Update current workspace
        if self.current_workspace != workspace:
            self.current_workspace = workspace
            print(f"Switched active workspace to '{workspace}'")
    def _load_guide(self, guide_name):
        """Load a guide from guides.txt"""
        debug_print(self.config, f"Loading guide: {guide_name}")

        try:
            with open("guides.txt", "r", encoding="utf-8") as file:
                content = file.read()

            import re
            pattern = rf"<{guide_name}>(.*?)</{guide_name}>"
            match = re.search(pattern, content, re.DOTALL)

            if match:
                guide_content = match.group(1).strip()
                print(f"Guide '{guide_name}' loaded:")
                print("-" * 40)
                print(guide_content)
                print("-" * 40)

                # Store the guide for later use
                self.config.set_guide(guide_name, guide_content)

                # Set as active guide
                self.active_guide = guide_name
                print(f"Guide '{guide_name}' set as active guide")
            else:
                print(f"Guide '{guide_name}' not found in guides.txt")
        except FileNotFoundError:
            print("guides.txt file not found")
        except Exception as e:
            print(f"Error loading guide: {str(e)}")

    # Clear Subcommand
    def _cmd_clear(self, args):
        """
        Handle the clear command to remove logs, vector databases, or cached data
        with improved structure and reusable utility functions
        """
        debug_print(self.config, f"Clear command with args: {args}")

        if not args:
            self.output_manager.print_formatted('feedback',
                                                "Usage: /clear [logs|vdb|cache|all] [workspace]",
                                                success=False)
            print("  logs    - Clear log files")
            print("  vdb     - Clear vector database files")
            print("  cache   - Clear cached data and intermediary files")
            print("  all     - Clear all of the above")
            return

        subcommand = args[0].lower()

        # Determine target workspace
        workspace = args[1] if len(args) > 1 else self.current_workspace

        # Command mapping
        clear_handlers = {
            'logs': self._clear_logs,
            'vdb': self._clear_vector_database,
            'cache': self._clear_cache,
            'all': self._clear_all
        }

        handler = clear_handlers.get(subcommand)

        if handler:
            handler(workspace)
        else:
            self.output_manager.print_formatted('feedback',
                                                f"Unknown clear subcommand: {subcommand}",
                                                success=False)
            print("Valid subcommands: logs, vdb, cache, all")
    def _clear_logs(self, workspace):
        """Clear log files for a workspace with enhanced feedback"""
        self.output_manager.print_formatted('header', f"Clearing Logs for '{workspace}'")

        # Use utility function to get and remove files
        log_files = self._get_files_by_patterns(
            os.path.join("logs", workspace),
            ["*.txt", "*.json", "*.md"]
        )

        if not log_files:
            self.output_manager.print_formatted('feedback',
                                                f"No log files found for workspace '{workspace}'",
                                                success=False)
            return

        # Ask for confirmation
        if not self._confirm_operation(f"clear {len(log_files)} log files"):
            return

        # Remove files with feedback
        removed = self._remove_files(log_files)

        self.output_manager.print_formatted('feedback',
                                            f"Cleared {removed} log files for workspace '{workspace}'",
                                            success=True)
    def _clear_vector_database(self, workspace):
        """Clear vector database files for a workspace"""
        self.output_manager.print_formatted('header', f"Clearing Vector Database for '{workspace}'")

        vector_dir = os.path.join("data", workspace, "vector_store")
        if not os.path.exists(vector_dir):
            self.output_manager.print_formatted('feedback',
                                                f"No vector database found for workspace '{workspace}'",
                                                success=False)
            return

        # Ask for confirmation with stronger warning
        if not self._confirm_operation(
                f"clear vector database for '{workspace}'",
                "This will require rebuilding indexes"
        ):
            return

        # Create backup
        backup_dir = self._create_backup(vector_dir)
        if backup_dir:
            self.output_manager.print_formatted('feedback',
                                                f"Created backup of vector database at: {backup_dir}",
                                                success=True)

        # Delete and recreate vector database directory
        try:
            import shutil
            shutil.rmtree(vector_dir)
            os.makedirs(vector_dir)  # Recreate empty directory
            self.output_manager.print_formatted('feedback',
                                                f"Cleared vector database for workspace '{workspace}'",
                                                success=True)
        except Exception as e:
            debug_print(self.config, f"Error clearing vector database: {str(e)}")
            self.output_manager.print_formatted('feedback',
                                                f"Error clearing vector database: {str(e)}",
                                                success=False)
    def _clear_cache(self, workspace):
        """Clear cached data and intermediary files"""
        self.output_manager.print_formatted('header', f"Clearing Cache for '{workspace}'")

        # Paths for cache files
        cache_patterns = [
            os.path.join("data", workspace, "*.json"),  # JSON cache files
            os.path.join("data", workspace, "*.cache"),  # Cache files
            os.path.join("data", workspace, "*.tmp")  # Temporary files
        ]

        # Gather all files matching patterns
        cache_files = []
        for pattern in cache_patterns:
            cache_files.extend(self._get_files_by_patterns("", [pattern]))

        if not cache_files:
            self.output_manager.print_formatted('feedback',
                                                f"No cache files found for workspace '{workspace}'",
                                                success=False)
            return

        # Ask for confirmation
        if not self._confirm_operation(f"clear {len(cache_files)} cache files"):
            return

        # Remove files with feedback
        removed = self._remove_files(cache_files)

        self.output_manager.print_formatted('feedback',
                                            f"Cleared {removed} cache files for workspace '{workspace}'",
                                            success=True)
    def _clear_all(self, workspace):
        """Clear all logs, vector database, and cache files"""
        self.output_manager.print_formatted('header', f"Clearing All Data for '{workspace}'")

        # Ask for confirmation with stronger warning
        if not self._confirm_operation(
                f"clear ALL data (logs, vector database, and cache) for '{workspace}'",
                "This is a destructive operation that cannot be undone"
        ):
            return

        # Clear each type of data
        self._clear_logs(workspace)
        self._clear_vector_database(workspace)
        self._clear_cache(workspace)

        self.output_manager.print_formatted('feedback',
                                            f"All clearing operations completed for workspace '{workspace}'",
                                            success=True)

    # Utility functions for file operations
    def _format_command(self, command_string):
        """Format a command string for display in logs and feedback"""
        if command_string.startswith('/'):
            tokens = command_string.split()
            command = tokens[0][1:]  # Remove leading slash
            args = tokens[1:] if len(tokens) > 1 else []

            if args:
                return f"{command} {' '.join(args)}"
            else:
                return command
        else:
            return command_string
    def _get_files_by_patterns(self, directory, patterns):
        """
        Get all files matching the given patterns

        Args:
            directory: Base directory to search
            patterns: List of glob patterns

        Returns:
            List of file paths
        """
        import glob

        if not os.path.exists(directory) and directory:
            return []

        files = []
        for pattern in patterns:
            # If directory is empty, just use the pattern as is
            if directory:
                path_pattern = os.path.join(directory, pattern)
            else:
                path_pattern = pattern

            files.extend(glob.glob(path_pattern))

        return files
    def _confirm_operation(self, operation, warning=None):
        """
        Ask for confirmation before performing destructive operations

        Args:
            operation: Description of the operation
            warning: Optional additional warning message

        Returns:
            Boolean indicating whether to proceed
        """
        prompt = f"Are you sure you want to {operation}? (y/n): "

        if warning:
            print(f"WARNING: {warning}")

        confirm = input(prompt).strip().lower()

        if confirm not in ('y', 'yes'):
            self.output_manager.print_formatted('feedback',
                                                "Operation cancelled",
                                                success=False)
            return False

        return True
    def _remove_files(self, file_list):
        """
        Remove a list of files with error handling

        Args:
            file_list: List of file paths to remove

        Returns:
            Number of files successfully removed
        """
        removed_count = 0

        for file_path in file_list:
            try:
                os.remove(file_path)
                removed_count += 1
                debug_print(self.config, f"Removed file: {file_path}")
            except Exception as e:
                debug_print(self.config, f"Error removing file {file_path}: {str(e)}")

        return removed_count
    def _create_backup(self, directory):
        """
        Create a timestamped backup of a directory

        Args:
            directory: Directory to back up

        Returns:
            Path to backup directory or None if failed
        """
        import datetime
        import shutil

        try:
            backup_dir = f"{directory}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copytree(directory, backup_dir)
            return backup_dir
        except Exception as e:
            debug_print(self.config, f"Error creating backup: {str(e)}")
            return None

    HELP_TOPICS = {
        "run": """
        Run Commands:
          /run themes [method]     - Run theme analysis
            Methods: all, nfm, net, key, lsa, cluster
              all    - Run all theme analysis methods
              nfm    - Named entity-based theme extraction
              net    - Content network analysis
              key    - Keyword-based theme identification
              lsa    - Latent semantic analysis
              cluster - Document clustering
    
          /run query             - Enter interactive query mode with context-aware responses
    
          /run grind             - Process files in workspace to create initial database
    
          /run update            - Update workspace with new or modified files
    
          /run scan [detailed]   - Scan workspace for new or updated files
            detailed - Show additional file information
    
          /run merge <source_ws> - Merge source workspace into current workspace
    
          /run sentiment [method] - Run sentiment analysis
            Methods: all, basic, advanced
              all     - Run all sentiment analysis methods
              basic   - Simple sentiment classification
              advanced - In-depth analysis with aspect extraction
    
          /run topics [method]    - Run topic modeling
            Methods: all, lda, nmf, cluster
              all    - Run all topic modeling methods
              lda    - Latent Dirichlet Allocation
              nmf    - Non-Negative Matrix Factorization
              cluster - Clustering-based topic modeling
    
        Examples:
          /run themes nfm        - Run named entity-based theme extraction
          /run sentiment advanced - Run advanced sentiment analysis
          /run topics lda         - Run LDA topic modeling
        """,
        "clear": """
        Clear Commands:
          /clear logs [workspace]    - Clear log files for a workspace
          /clear vdb [workspace]     - Clear vector database files for a workspace
          /clear cache [workspace]   - Clear cached data and intermediary files
          /clear all [workspace]     - Clear all logs, vector database, and cache files
    
        Notes:
          - If workspace is not specified, the current workspace is used
          - Clearing vector database will require rebuilding indexes
          - A backup is created before clearing the vector database
          - All clear operations require confirmation
        """,
        "load": """
        Load Commands:
          /load ws <workspace>  - Load or create a workspace
    
        Notes:
          - If the workspace doesn't exist, you'll be prompted to create it
          - Loading a workspace sets it as the current active workspace
          - All commands will operate on the current workspace
        """,
        "config": """
        Config Commands:
          /config llm <model>                - Set LLM model (e.g., mistral, llama2)
          /config embed <model>              - Set embedding model (e.g., multilingual-e5, mixbread)
          /config storage <backend>          - Set storage backend (llama_index, haystack, or chroma)
          /config kval <n>                   - Set k value for retrieval (number of docs to return)
          /config debug [on|off]             - Enable or disable debug mode
          /config output [txt|md|json]       - Set output format for saved files
          /config guide <guide>              - Set guide from guides.txt
    
        Examples:
          /config llm mistral                - Set LLM model to mistral
          /config embed multilingual-e5      - Set embedding model to multilingual-e5
          /config kval 5                     - Set retrieval to return 5 documents
        """,
        "save": """
        Save Commands:
          /save start                - Start saving session (all commands and outputs)
          /save stop                 - Stop saving session and write to file
          /save buffer               - Save last command output to file
    
        Notes:
          - Session logs are saved to logs/<workspace>/ directory
          - Output format depends on your config.output_format setting
          - The buffer contains the most recent command output
        """,
        "show": """
        Show Commands:
          /show status               - Show system status (current settings)
          /show cuda                 - Show CUDA status and GPU information
          /show config               - Show detailed configuration settings
          /show ws                   - Show workspace details and statistics
          /show files                - Show files in current workspace
          /show guide                - Show available guides from guides.txt
    
        Examples:
          /show status               - View current workspace and model settings
          /show files                - List all files in the current workspace
        """,
        "inspect": """
        Inspect Commands:
          /inspect workspace [ws]         - Inspect workspace metadata and vector store
          /inspect ws [ws]                - Alias for workspace inspect
          /inspect documents [ws] [limit] - Dump document content in workspace
            Optional: Specify number of documents to show (default: 5)
          /inspect vectorstore [ws]       - Dump vector store metadata
          /inspect vdb [ws]               - Alias for vectorstore inspect
          /inspect query [ws] [query]     - Test query pipeline with optional test query
          /inspect rebuild [ws]           - Rebuild vector store from existing documents
          /inspect fix [ws]               - Fix common vector store issues
          /inspect metadata [ws] [query]  - Inspect raw metadata returned from vector store
          /inspect migrate [ws]           - Fix inconsistent vector store naming
    
        Examples:
          /inspect documents default 10   - Show content of 10 documents in default workspace
          /inspect query research         - Test query pipeline with the term "research"
        """,
        "explain": """
        Interpretation Commands:
          /explain themes [question]    - Get LLM interpretation of theme analysis
          /explain topics [question]    - Get LLM interpretation of topic modeling 
          /explain sentiment [question] - Get LLM interpretation of sentiment analysis
          /explain session [question]   - Get LLM interpretation of query session
    
        Examples:
          /explain themes What are the most significant themes?
          /explain topics How do the topics relate to each other?
          /explain sentiment What emotions are most prominent?
          /explain session What were the main research directions?
        """
    }

    HELP_OVERVIEW = """
    Available Commands:
    
      Run Modes:
        /run grind          - Process files in workspace to create initial database
        /run update         - Update workspace with new or modified files
        /run scan           - Scan workspace for new or updated files
        /run merge          - Merge workspaces
        /run sentiment      - Run sentiment analysis
        /run topics         - Run topic modeling
        /run themes         - Run theme analysis
        /run query          - Enter interactive query mode
    
      Interpretation:
        /explain            - Get LLM interpretation of analysis
    
      Operations:
        /load               - Load workspace
        /save               - Start/stop saving session or buffer
        /config             - Set runtime configuration
        /show               - Show active information and data sources
        /clear              - Clear logs, vector database, or cache files
        /inspect            - Check/verify status of data stores 
        /quit, /exit        - Exit the application
        /help [command]     - Display help information
    
      Aliases:
        /q - /quit, /c - /config, /l - /load, /r - /run
        /s - /save, /h - /help, /i - /inspect, /e - /explain
        /cl - /clear
    """
