"""
Command line interface for the document analysis toolkit
"""

import os
import shlex
import sys
from core.engine.logging import debug, error, warning, info, trace
from core.engine.storage import StorageManager, VectorStoreInspector
from core.engine.output import OutputManager
import datetime
from core.engine.interpreter import AnalysisInterpreter
from core.engine.utils import Colors


class CommandProcessor:
    """Command line interface processor"""

    def __init__(self, config):
        """Initialize CommandProcessor with configuration"""
        self.config = config
        self.running = True
        self.current_workspace = config.get('workspace.default')
        self.hpc_mode = config.get('system.hpc_mode', False)
        self.batch_mode = config.get('system.batch_mode', False)

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

        # Create the batch processor
        from core.engine.batch import BatchProcessor
        self.batch_processor = BatchProcessor(self)

        info("CommandProcessor initialized")

    def start(self):
        """Start the command processing loop"""
        debug("Starting command processing loop")
        info(f"Monkey v{self.config.get_version()} initialized. Type /help for available commands.")

        try:
            while self.running:
                # Show workspace and current LLM model in the prompt
                llm_model = self.config.get('llm.default_model')

                # Add [query] indicator when in query mode
                if hasattr(self, 'query_engine') and self.query_engine.is_active():
                    prompt = f"[{self.current_workspace}][{llm_model}][query]> "
                else:
                    prompt = f"[{self.current_workspace}][{llm_model}]> "

                try:
                    user_input = input(prompt)
                    if user_input.strip():
                        self.process_command(user_input)
                except KeyboardInterrupt:
                    print("\nUse /quit or /exit to exit")
                except EOFError:
                    print("\nExiting...")
                    self.running = False

        except KeyboardInterrupt:
            print("\nExiting...")
        finally:
            debug(self.config, "Command processing loop ended")

    def process_batch_file(self, batch_file):
        """
        Process commands from a batch file

        Args:
            batch_file (str): Path to the batch file

        Returns:
            bool: Success flag
        """
        return self.batch_processor.process_file(batch_file)

    def _handle_system_command(self, command, args):
        """Handle system commands"""
        debug(self.config, f"Handling system command: {command} with args: {args}")

        # Handle aliases
        command = self._resolve_alias(command)

        # Special handling for exit/quit in query mode
        if (command == 'exit' or command == 'quit') and hasattr(self, 'query_engine') and self.query_engine.is_active():
            # Just exit query mode, not the whole application
            debug(self.config, "Exit/quit command in query mode - deactivating query mode")
            self.query_engine.deactivate()
            return

        # Process commands
        if command == 'quit' or command == 'exit':
            self._cmd_quit()
        elif command == 'help':
            self._cmd_help(args)
        elif command == 'show':
            self._cmd_show(args)
        elif command == 'run':
            self._cmd_run(args)
        elif command == 'load':
            self._cmd_load(args)
        elif command == 'save':
            self._cmd_save(args)
        elif command == 'config':
            self._cmd_config(args)
        elif command == 'inspect':
            self._cmd_inspect(args)
        elif command == 'explain':
            self._cmd_explain(args)
        elif command == 'clear':
            self._cmd_clear(args)
        else:
            print(f"Unknown command: /{command}")

    def _resolve_alias(self, command):
        """Resolve command aliases to full commands"""
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

    def _cmd_config(self, args):
        """Handle the config command"""
        debug(self.config, f"Config command with args: {args}")

        if not args:
            self._show_config()
            return

        # Get the subcommand
        subcommand = args[0].lower()

        if subcommand == 'llm':
            if len(args) < 2:
                print(f"Current LLM model: {self.config.get('llm.default_model')}")
                return
            self.config.set('llm.default_model', args[1])
            print(f"LLM model set to: {args[1]}")

        elif subcommand == 'embed':
            if len(args) < 2:
                print(f"Current embedding model: {self.config.get('embedding.default_model')}")
                print("Available models: multilingual-e5, mixbread, bge, jina-zh")
                return
            self.config.set('embedding.default_model', args[1])
            print(f"Embedding model set to: {args[1]}")

        elif subcommand == 'storage':
            if len(args) < 2:
                print(f"Current storage backend: {self.config.get('storage.vector_store')}")
                print("Available backends: llama_index, haystack, chroma")
                return
            self.config.set('storage.vector_store', args[1])
            print(f"Storage backend set to: {args[1]}")

        elif subcommand == 'kval':
            if len(args) < 2:
                print(f"Current k value: {self.config.get('query.k_value')}")
                return
            try:
                k_value = int(args[1])
                self.config.set('query.k_value', k_value)
                print(f"K value set to: {k_value}")
            except ValueError:
                print(f"Invalid k value: {args[1]}")

        elif subcommand == 'debug':
            if len(args) < 2:
                print(f"Current debug level: {self.config.get('system.debug_level')}")
                return
            debug_value = args[1].lower()
            if debug_value in ['on', 'true', 'yes', '1']:
                self.config.set('system.debug_level', 'debug')
                print("Debug mode enabled")
            elif debug_value in ['off', 'false', 'no', '0']:
                self.config.set('system.debug_level', 'info')
                print("Debug mode disabled")
            else:
                # Try to set explicit level
                self.config.set('system.debug_level', debug_value)
                print(f"Debug level set to: {debug_value}")

        elif subcommand == 'output':
            if len(args) < 2:
                print(f"Current output format: {self.config.get('system.output_format')}")
                return
            output_format = args[1].lower()
            if output_format in ['txt', 'json']:
                self.config.set('system.output_format', output_format)
                print(f"Output format set to: {output_format}")
            else:
                print(f"Unsupported output format: {output_format}")
                print("Supported formats: txt, json")

        elif subcommand == 'guide':
            if len(args) < 2:
                print(f"Current active guide: {self.active_guide or 'None'}")
                return
            guide_name = args[1]
            self._load_guide(guide_name)

        elif subcommand == 'device':
            if len(args) < 2:
                print(f"Current device: {self.config.get('hardware.device')}")
                return
            device = args[1].lower()
            if device in ['auto', 'cpu', 'cuda', 'mps']:
                self.config.set('hardware.device', device)
                print(f"Device set to: {device}")
            else:
                print(f"Unsupported device: {device}")
                print("Supported devices: auto, cpu, cuda, mps")

        elif subcommand == 'batch':
            if len(args) < 2:
                print(f"Current batch settings:")
                print(f"  Exit on error: {self.config.get('batch.exit_on_error')}")
                print(f"  Timeout: {self.config.get('batch.timeout')} seconds")
                return
            batch_setting = args[1].lower()
            if batch_setting == 'exit_on_error':
                if len(args) < 3:
                    print(f"Current exit_on_error setting: {self.config.get('batch.exit_on_error')}")
                    return
                value = args[2].lower() in ['true', 'yes', '1', 't', 'y']
                self.config.set('batch.exit_on_error', value)
                print(f"Batch exit_on_error set to: {value}")
            elif batch_setting == 'timeout':
                if len(args) < 3:
                    print(f"Current timeout setting: {self.config.get('batch.timeout')} seconds")
                    return
                try:
                    timeout = int(args[2])
                    self.config.set('batch.timeout', timeout)
                    print(f"Batch timeout set to: {timeout} seconds")
                except ValueError:
                    print(f"Invalid timeout value: {args[2]}")
            else:
                print(f"Unknown batch setting: {batch_setting}")
                print("Available settings: exit_on_error, timeout")

        elif subcommand == 'hpc':
            if len(args) < 2:
                print(f"Current HPC mode: {self.config.get('system.hpc_mode')}")
                return
            hpc_mode = args[1].lower() in ['true', 'yes', '1', 't', 'y', 'on']
            self.config.set('system.hpc_mode', hpc_mode)
            # Update the instance variable too
            self.hpc_mode = hpc_mode
            print(f"HPC mode set to: {hpc_mode}")

        elif subcommand == 'export':
            # Export config to environment variables
            self.config.export_to_env()
            print("Configuration exported to environment variables")

        elif subcommand == 'log':
            # Handle log configuration
            self._config_log(args[1:] if len(args) > 1 else [])

        elif subcommand == 'reset':
            # Reset to defaults
            if len(args) < 2:
                print("Usage: /config reset <section.key> or 'all'")
                return

            target = args[1].lower()
            if target == 'all':
                # Re-initialize with defaults
                self.config._ensure_defaults()
                print("All configuration reset to defaults")
            else:
                # Try to reset specific section.key
                sections = target.split('.')
                if len(sections) != 2:
                    print("Invalid format. Use 'section.key' (e.g., 'system.debug_level')")
                    return

                section, key = sections
                if section not in self.config.config:
                    print(f"Section '{section}' not found in configuration")
                    return

                if key not in self.config.config[section]:
                    print(f"Key '{key}' not found in section '{section}'")
                    return

                # Reset using defaults
                self.config._ensure_defaults()
                default_value = self.config.config[section][key]

                # Set it back to trigger proper update behavior
                self.config.set(target, default_value)
                print(f"Reset {target} to default: {default_value}")

        else:
            print(f"Unknown config subcommand: {subcommand}")
            print(
                "Available subcommands: llm, embed, storage, kval, debug, output, guide, device, batch, hpc, export, log, reset")

    def _config_log(self, args):
        """Handle log configuration as part of the config command"""
        debug(self.config, f"Log config with args: {args}")

        from core.engine.logging import LogManager

        if not args:
            print("Usage: /config log [file|console|both|status]")
            return

        subcommand = args[0].lower()

        if subcommand == 'file':
            if len(args) < 2:
                print("Usage: /config log file <filename>")
                return

            log_file = args[1]
            # Ensure directory exists
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            # Redirect all logs to file
            LogManager.redirect_all_logs(log_file)
            print(f"All logs redirected to: {log_file}")

        elif subcommand == 'console':
            # Reset logging to console only
            from core.engine.logging import LogManager

            # Reconfigure logging
            debug_level = self.config.get('system.debug_level', 'info')
            LogManager.configure()
            LogManager.set_level(debug_level)

            print("Logging restored to console only")

        elif subcommand == 'both':
            if len(args) < 2:
                print("Usage: /config log both <filename>")
                return

            log_file = args[1]

            # Configure logging to both console and file
            debug_level = self.config.get('system.debug_level', 'info')
            LogManager.configure()
            LogManager.set_level(debug_level)
            LogManager.add_file_handler(log_file)

            print(f"Logging to both console and: {log_file}")

        elif subcommand == 'status':
            # Show current logging configuration
            log_file = LogManager.get_log_file_path()

            if log_file:
                print(f"Currently logging to file: {log_file}")
            else:
                print("Currently logging to console only")

            # Show debug level
            debug_level = self.config.get('system.debug_level', 'info')
            print(f"Current debug level: {debug_level}")

        else:
            print(f"Unknown log subcommand: {subcommand}")
            print("Available subcommands: file, console, both, status")

    def _show_specific_help(self, topic):
        """Show help for a specific topic"""
        if topic == 'config':
            print("""
            Config Commands:
              /config llm <model>                - Set LLM model (e.g., mistral, llama2)
              /config embed <model>              - Set embedding model (e.g., multilingual-e5, mixbread)
              /config storage <backend>          - Set storage backend (llama_index, haystack, or chroma)
              /config kval <n>                   - Set k value for retrieval (number of docs to return)
              /config debug [on|off]             - Enable or disable debug mode
              /config output [txt|md|json]       - Set output format for saved files
              /config guide <guide>              - Set guide from guides.txt
              /config log <option>               - Control logging to file (see below)

              Log control options:
                /config log file <filename>      - Redirect all logging to a file
                /config log console              - Restore logging to console
                /config log both <filename>      - Log to both console and file
                /config log status               - Show current logging status

            Examples:
              /config llm mistral                - Set LLM model to mistral
              /config embed multilingual-e5      - Set embedding model to multilingual-e5
              /config kval 5                     - Set retrieval to return 5 documents
              /config log file logs/output.log   - Redirect all logs to file
            """)

    def _cmd_quit(self):
        """Handle the quit command"""
        debug(self.config, "Quit command received")
        self.running = False

    def _cmd_help(self, args):
        """Display help information"""
        debug(self.config, f"Help command with args: {args}")

        if not args:
            print("""
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
        """)
        elif args[0] in ['run', 'load', 'config', 'save', 'show', 'inspect', 'explain', 'clear']:
            self._show_specific_help(args[0])
        else:
            print(f"No specific help available for '{args[0]}'")

    def _cmd_show(self, args):
        """Handle the show command"""
        debug(self.config, f"Show command with args: {args}")

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

    def _show_status(self):
        """Show system status with compact colored formatting"""
        debug(self.config, "Showing system status")

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
        debug(self.config, "Showing workspace details")

        self.output_manager.print_formatted('header', "WORKSPACES")

        for ws in self.loaded_workspaces:
            if ws == self.current_workspace:
                self.output_manager.print_formatted('list', f"{ws} (active)", indent=2)
            else:
                self.output_manager.print_formatted('list', ws, indent=2)

        # Get additional workspace stats if available
        if hasattr(self, 'storage_manager'):
            stats = self.storage_manager.get_workspace_stats(self.current_workspace)
            if stats:
                self.output_manager.print_formatted('subheader', f"Details for {self.current_workspace}")

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
        debug(self.config, "Showing workspace files")

        import os
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
        else:
            # Print compact file list with colors
            from core.engine.utils import Colors
            print(
                f"\n  {Colors.CYAN}Filename{Colors.RESET}  {Colors.CYAN}Size{Colors.RESET}  {Colors.CYAN}Modified{Colors.RESET}")

            for file_path, size, mod_time in sorted(files):
                print(f"  {Colors.BRIGHT_WHITE}{file_path}{Colors.RESET}  {self._format_size(size)}  {mod_time}")

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

    def process_command(self, command_string):
        """
        Process a command string

        Args:
            command_string (str): The command to process
        """
        debug(self.config, f"Processing command: {command_string}")

        # Echo the command to the output manager
        self.output_manager.print_formatted('command', command_string)

        # Skip empty commands
        if not command_string.strip():
            return

        # Parse the command
        try:
            tokens = shlex.split(command_string)
            command = tokens[0].lower()
            args = tokens[1:]
        except Exception as e:
            self.output_manager.print_formatted('feedback', f"Error parsing command: {str(e)}", success=False)
            return

        # Handle commands
        if command.startswith('/'):
            # Always process system commands, even in query mode
            self._handle_system_command(command[1:], args)
        else:
            # Treat as a query in interactive mode
            if hasattr(self, 'query_engine') and self.query_engine.is_active():
                self.query_engine.process_query(command_string)
            else:
                self.output_manager.print_formatted('feedback',
                                                    "Not in query mode. Use '/run query' to enter interactive query mode.",
                                                    success=False)

    def _cmd_explain(self, args):
        """Handle the explain command for analysis interpretation"""
        debug(self.config, f"Explain command with args: {args}")

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

    def _cmd_clear(self, args):
        """
        Handle the clear command to remove logs, vector databases, or cached data

        Args:
            args (List[str]): Command arguments
        """
        debug(self.config, f"Clear command with args: {args}")

        if not args:
            print("Usage: /clear [logs|vdb|cache|all] [workspace]")
            print("  logs    - Clear log files")
            print("  vdb     - Clear vector database files")
            print("  cache   - Clear cached data and intermediary files")
            print("  all     - Clear all of the above")
            return

        subcommand = args[0].lower()

        # Determine target workspace
        workspace = args[1] if len(args) > 1 else self.current_workspace

        if subcommand == 'logs':
            self._clear_logs(workspace)
        elif subcommand == 'vdb':
            self._clear_vector_database(workspace)
        elif subcommand == 'cache':
            self._clear_cache(workspace)
        elif subcommand == 'all':
            self._clear_logs(workspace)
            self._clear_vector_database(workspace)
            self._clear_cache(workspace)
        else:
            print(f"Unknown clear subcommand: {subcommand}")
            print("Valid subcommands: logs, vdb, cache, all")

    def _clear_logs(self, workspace):
        """Clear log files for a workspace"""
        import os
        import glob

        logs_dir = os.path.join("logs", workspace)
        if not os.path.exists(logs_dir):
            print(f"No logs directory found for workspace '{workspace}'")
            return

        # Ask for confirmation
        confirm = input(f"Are you sure you want to clear all logs for workspace '{workspace}'? (y/n): ").strip().lower()
        if confirm != 'y' and confirm != 'yes':
            print("Operation cancelled")
            return

        # Delete log files
        log_files = glob.glob(os.path.join(logs_dir, "*.txt"))
        log_files.extend(glob.glob(os.path.join(logs_dir, "*.json")))
        log_files.extend(glob.glob(os.path.join(logs_dir, "*.md")))

        if not log_files:
            print(f"No log files found for workspace '{workspace}'")
            return

        for file in log_files:
            try:
                os.remove(file)
                debug(self.config, f"Removed log file: {file}")
            except Exception as e:
                debug(self.config, f"Error removing file {file}: {str(e)}")

        print(f"Cleared {len(log_files)} log files for workspace '{workspace}'")

    def _clear_vector_database(self, workspace):
        """Clear vector database files for a workspace"""
        import os
        import shutil

        vector_dir = os.path.join("data", workspace, "vector_store")
        if not os.path.exists(vector_dir):
            print(f"No vector database found for workspace '{workspace}'")
            return

        # Ask for confirmation
        confirm = input(
            f"Are you sure you want to clear the vector database for workspace '{workspace}'? This will require rebuilding indexes. (y/n): ").strip().lower()
        if confirm != 'y' and confirm != 'yes':
            print("Operation cancelled")
            return

        # Create backup
        import datetime
        backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copytree(vector_dir, backup_dir)
            print(f"Created backup of vector database at: {backup_dir}")
        except Exception as e:
            debug(self.config, f"Error creating backup: {str(e)}")
            # Continue with deletion even if backup fails

        # Delete vector database
        try:
            shutil.rmtree(vector_dir)
            os.makedirs(vector_dir)  # Recreate empty directory
            print(f"Cleared vector database for workspace '{workspace}'")
        except Exception as e:
            debug(self.config, f"Error clearing vector database: {str(e)}")
            print(f"Error clearing vector database: {str(e)}")

    def _clear_cache(self, workspace):
        """Clear cached data and intermediary files"""
        import os
        import glob

        # Paths for cache files
        cache_paths = [
            os.path.join("data", workspace, "*.json"),  # JSON cache files
            os.path.join("data", workspace, "*.cache"),  # Cache files
            os.path.join("data", workspace, "*.tmp")  # Temporary files
        ]

        # Ask for confirmation
        confirm = input(
            f"Are you sure you want to clear cache files for workspace '{workspace}'? (y/n): ").strip().lower()
        if confirm != 'y' and confirm != 'yes':
            print("Operation cancelled")
            return

        # Count cleared files
        cleared_count = 0

        # Remove cache files
        for path_pattern in cache_paths:
            files = glob.glob(path_pattern)
            for file in files:
                try:
                    os.remove(file)
                    cleared_count += 1
                    debug(self.config, f"Removed cache file: {file}")
                except Exception as e:
                    debug(self.config, f"Error removing file {file}: {str(e)}")

        print(f"Cleared {cleared_count} cache files for workspace '{workspace}'")

    def _cmd_load(self, args):
        """Handle the load command"""
        debug(self.config, f"Load command with args: {args}")

        if not args:
            print("Usage: /load ws <workspace>")
            return

        # Check if loading workspace
        if args[0].lower() == 'ws':
            if len(args) < 2:
                print("Usage: /load ws <workspace>")
                return
            workspace = args[1]
            self._load_workspace(workspace)
        else:
            print(f"Unknown load target: {args[0]}")
            print("Usage: /load ws <workspace>")

    def _load_workspace(self, workspace):
        """
        Load a workspace with confirmation for new workspaces

        Args:
            workspace (str): The workspace to load
        """
        debug(self.config, f"Loading workspace: {workspace}")

        # Check if workspace directories exist
        from pathlib import Path
        data_dir = Path("data") / workspace
        body_dir = Path("body") / workspace

        # Check if this is a new workspace
        is_new_workspace = not data_dir.exists() and not body_dir.exists()

        if is_new_workspace:
            # Ask for confirmation before creating a new workspace
            confirm = input(f"Workspace '{workspace}' does not exist. Create it? (y/n): ").strip().lower()
            if confirm != 'y' and confirm != 'yes':
                warning(f"Cancelled creation of workspace '{workspace}'")
                return

        # Create necessary directories
        for directory in [data_dir, body_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
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

    def _load_guide(self, guide_name):
        """Load a guide from guides.txt"""
        debug(self.config, f"Loading guide: {guide_name}")

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

    def _cmd_inspect(self, args):
        """Handle the inspect command"""
        debug(self.config, f"Inspect command with args: {args}")

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
            # Check for LlamaIndex specific inspection
            if self.config.get('storage.vector_store') == 'llama_index':
                try:
                    from core.connectors.llama_index_connector import LlamaIndexConnector
                    llama_connector = LlamaIndexConnector(self.config)
                    llama_connector.inspect_index_store(workspace)
                except Exception as e:
                    error(f"Error inspecting LlamaIndex store: {str(e)}")
        elif subcommand == 'query':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            query = args[2] if len(args) > 2 else "test"
            self.vector_store_inspector.test_query_pipeline(workspace, query)
        elif subcommand == 'rebuild':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            self.vector_store_inspector.rebuild_vector_store(workspace)
        elif subcommand == 'fix':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            warning(f"Attempting to fix vector store issues for workspace '{workspace}'...")
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

    def _cmd_run(self, args):
        """Handle the run command"""
        debug(self.config, f"Run command with args: {args}")

        if not args:
            print("Usage: /run [themes|query|grind|update|scan|merge|sentiment|topic] [options]")
            return

        subcommand = args[0].lower()

        if subcommand == 'themes':
            method = args[1] if len(args) > 1 else 'all'
            self.theme_analyzer.analyze(self.current_workspace, method)

        elif subcommand == 'query':
            # Check if this is a one-time query (args contains the query text)
            if len(args) > 1:
                # Extract the query text (everything after 'query')
                query_text = ' '.join(args[1:])

                # Activate query engine
                if not self.query_engine.is_active():
                    if not self.query_engine.activate(self.current_workspace):
                        print(f"Failed to activate query engine for workspace '{self.current_workspace}'")
                        return

                # Process the one-time query
                response = self.query_engine.process_one_time_query(query_text)

                # Print the response
                print("\nRESPONSE:")
                print(response)
                print()
            else:
                # Enter interactive query mode
                if self.query_engine.activate(self.current_workspace):
                    self.query_engine.enter_interactive_mode()

        elif subcommand == 'grind':
            self.file_processor.process_workspace(self.current_workspace)

        elif subcommand == 'update':
            self.file_processor.update_workspace(self.current_workspace)

        elif subcommand == 'scan':
            detailed = len(args) > 1 and args[1].lower() == 'detailed'
            self.file_processor.scan_workspace(self.current_workspace, detailed)

        elif subcommand == 'merge':
            if len(args) < 2:
                print("Usage: /run merge <source_workspace>")
                return
            source_workspace = args[1]
            self.vector_store_merger.merge(source_workspace, self.current_workspace)

        elif subcommand == 'sentiment':
            method = args[1] if len(args) > 1 else 'all'
            self.sentiment_analyzer.analyze(self.current_workspace, method)

        elif subcommand == 'topics':
            method = args[1] if len(args) > 1 else 'all'
            self.topic_modeler.analyze(self.current_workspace, method)

        else:
            print(f"Unknown run subcommand: {subcommand}")

    def _cmd_explain(self, args):
        """Handle the explain command for analysis interpretation"""
        debug(self.config, f"Explain command with args: {args}")

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

    def _cmd_clear(self, args):
        """
        Handle the clear command to remove logs, vector databases, or cached data

        Args:
            args (List[str]): Command arguments
        """
        debug(self.config, f"Clear command with args: {args}")

        if not args:
            print("Usage: /clear [logs|vdb|cache|all] [workspace]")
            print("  logs    - Clear log files")
            print("  vdb     - Clear vector database files")
            print("  cache   - Clear cached data and intermediary files")
            print("  all     - Clear all of the above")
            return

        subcommand = args[0].lower()

        # Determine target workspace
        workspace = args[1] if len(args) > 1 else self.current_workspace

        if subcommand == 'logs':
            self._clear_logs(workspace)
        elif subcommand == 'vdb':
            self._clear_vector_database(workspace)
        elif subcommand == 'cache':
            self._clear_cache(workspace)
        elif subcommand == 'all':
            self._clear_logs(workspace)
            self._clear_vector_database(workspace)
            self._clear_cache(workspace)
        else:
            print(f"Unknown clear subcommand: {subcommand}")
            print("Valid subcommands: logs, vdb, cache, all")

    def _clear_logs(self, workspace):
        """Clear log files for a workspace"""
        import os
        import glob

        logs_dir = os.path.join("logs", workspace)
        if not os.path.exists(logs_dir):
            print(f"No logs directory found for workspace '{workspace}'")
            return

        # Ask for confirmation
        confirm = input(f"Are you sure you want to clear all logs for workspace '{workspace}'? (y/n): ").strip().lower()
        if confirm != 'y' and confirm != 'yes':
            print("Operation cancelled")
            return

        # Delete log files
        log_files = glob.glob(os.path.join(logs_dir, "*.txt"))
        log_files.extend(glob.glob(os.path.join(logs_dir, "*.json")))
        log_files.extend(glob.glob(os.path.join(logs_dir, "*.md")))

        if not log_files:
            print(f"No log files found for workspace '{workspace}'")
            return

        for file in log_files:
            try:
                os.remove(file)
                debug(self.config, f"Removed log file: {file}")
            except Exception as e:
                debug(self.config, f"Error removing file {file}: {str(e)}")

        print(f"Cleared {len(log_files)} log files for workspace '{workspace}'")

    def _clear_vector_database(self, workspace):
        """Clear vector database files for a workspace"""
        import os
        import shutil
        import datetime

        vector_dir = os.path.join("data", workspace, "vector_store")
        if not os.path.exists(vector_dir):
            print(f"No vector database found for workspace '{workspace}'")
            return

        # Ask for confirmation
        confirm = input(
            f"Are you sure you want to clear the vector database for workspace '{workspace}'? This will require rebuilding indexes. (y/n): ").strip().lower()
        if confirm != 'y' and confirm != 'yes':
            print("Operation cancelled")
            return

        # Create backup
        backup_dir = f"{vector_dir}_backup_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            shutil.copytree(vector_dir, backup_dir)
            print(f"Created backup of vector database at: {backup_dir}")
        except Exception as e:
            debug(self.config, f"Error creating backup: {str(e)}")
            # Continue with deletion even if backup fails

        # Delete vector database
        try:
            shutil.rmtree(vector_dir)
            os.makedirs(vector_dir)  # Recreate empty directory
            print(f"Cleared vector database for workspace '{workspace}'")
        except Exception as e:
            debug(self.config, f"Error clearing vector database: {str(e)}")
            print(f"Error clearing vector database: {str(e)}")

    def _clear_cache(self, workspace):
        """Clear cached data and intermediary files"""
        import os
        import glob

        # Paths for cache files
        cache_paths = [
            os.path.join("data", workspace, "*.json"),  # JSON cache files
            os.path.join("data", workspace, "*.cache"),  # Cache files
            os.path.join("data", workspace, "*.tmp")  # Temporary files
        ]

        # Ask for confirmation
        confirm = input(
            f"Are you sure you want to clear cache files for workspace '{workspace}'? (y/n): ").strip().lower()
        if confirm != 'y' and confirm != 'yes':
            print("Operation cancelled")
            return

        # Count cleared files
        cleared_count = 0

        # Remove cache files
        for path_pattern in cache_paths:
            files = glob.glob(path_pattern)
            for file in files:
                try:
                    os.remove(file)
                    cleared_count += 1
                    debug(self.config, f"Removed cache file: {file}")
                except Exception as e:
                    debug(self.config, f"Error removing file {file}: {str(e)}")

        print(f"Cleared {cleared_count} cache files for workspace '{workspace}'")

    def _cmd_save(self, args):
        """Handle the save command"""
        debug(self.config, f"Save command with args: {args}")

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
        debug(self.config, "Starting session saving")

        if self.saving_session:
            print("Session saving is already active")
            return

        self.saving_session = True
        self.output_manager.start_session_saving(self.current_workspace)
        print("Session saving started")

    def _stop_saving(self):
        """Stop saving session output"""
        debug(self.config, "Stopping session saving")

        if not self.saving_session:
            print("Session saving is not active")
            return

        self.saving_session = False
        filepath = self.output_manager.stop_session_saving()
        print(f"Session saving stopped. Output saved to: {filepath}")

    def _save_buffer(self):
        """Save last command output buffer"""
        debug(self.config, "Saving output buffer")

        filepath = self.output_manager.save_buffer(self.current_workspace)
        if filepath:
            print(f"Buffer saved to: {filepath}")
        else:
            print("No buffer to save")

    def _show_config(self):
        """Show configuration details"""
        debug(self.config, "Showing configuration")
        self.config.display()

    def _show_cuda(self):
        """Show CUDA status"""
        debug(self.config, "Showing CUDA status")
        from core.engine.cuda import check_cuda_status
        check_cuda_status(self.config)

    def _show_workspace(self):
        """Show workspace details with compact colored formatting"""
        debug(self.config, "Showing workspace details")

        self.output_manager.print_formatted('header', "WORKSPACES")

        for ws in self.loaded_workspaces:
            if ws == self.current_workspace:
                self.output_manager.print_formatted('list', f"{ws} (active)", indent=2)
            else:
                self.output_manager.print_formatted('list', ws, indent=2)

        # Get additional workspace stats if available
        if hasattr(self, 'storage_manager'):
            stats = self.storage_manager.get_workspace_stats(self.current_workspace)
            if stats:
                self.output_manager.print_formatted('subheader', f"Details for {self.current_workspace}")

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
        debug(self.config, "Showing workspace files")

        import os
        import datetime
        from core.engine.utils import Colors, format_size

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
        else:
            # Print compact file list with colors
            print(
                f"\n  {Colors.CYAN}Filename{Colors.RESET}  {Colors.CYAN}Size{Colors.RESET}  {Colors.CYAN}Modified{Colors.RESET}")

            for file_path, size, mod_time in sorted(files):
                print(f"  {Colors.BRIGHT_WHITE}{file_path}{Colors.RESET}  {format_size(size)}  {mod_time}")

    def _cmd_config(self, args):
        """Handle the config command"""
        debug(self.config, f"Config command with args: {args}")

        if not args:
            self._show_config()
            return

        # Get the subcommand
        subcommand = args[0].lower()

        if subcommand == 'llm':
            if len(args) < 2:
                print(f"Current LLM model: {self.config.get('llm.default_model')}")
                return
            self.config.set('llm.default_model', args[1])
            print(f"LLM model set to: {args[1]}")

        elif subcommand == 'embed':
            if len(args) < 2:
                print(f"Current embedding model: {self.config.get('embedding.default_model')}")
                print("Available models: multilingual-e5, mixbread, bge, jina-zh")
                return
            self.config.set('embedding.default_model', args[1])
            print(f"Embedding model set to: {args[1]}")

        elif subcommand == 'storage':
            if len(args) < 2:
                print(f"Current storage backend: {self.config.get('storage.vector_store')}")
                print("Available backends: llama_index, haystack, chroma")
                return
            self.config.set('storage.vector_store', args[1])
            print(f"Storage backend set to: {args[1]}")

        elif subcommand == 'kval':
            if len(args) < 2:
                print(f"Current k value: {self.config.get('query.k_value')}")
                return
            try:
                k_value = int(args[1])
                self.config.set('query.k_value', k_value)
                print(f"K value set to: {k_value}")
            except ValueError:
                print(f"Invalid k value: {args[1]}")

        elif subcommand == 'debug':
            if len(args) < 2:
                print(f"Current debug level: {self.config.get('system.debug_level')}")
                return
            debug_value = args[1].lower()
            if debug_value in ['on', 'true', 'yes', '1']:
                self.config.set('system.debug_level', 'debug')
                print("Debug mode enabled")
            elif debug_value in ['off', 'false', 'no', '0']:
                self.config.set('system.debug_level', 'info')
                print("Debug mode disabled")
            else:
                # Try to set explicit level
                self.config.set('system.debug_level', debug_value)
                print(f"Debug level set to: {debug_value}")

        elif subcommand == 'output':
            if len(args) < 2:
                print(f"Current output format: {self.config.get('system.output_format')}")
                return
            output_format = args[1].lower()
            if output_format in ['txt', 'json']:
                self.config.set('system.output_format', output_format)
                print(f"Output format set to: {output_format}")
            else:
                print(f"Unsupported output format: {output_format}")
                print("Supported formats: txt, json")

        elif subcommand == 'guide':
            if len(args) < 2:
                print(f"Current active guide: {self.active_guide or 'None'}")
                return
            guide_name = args[1]
            self._load_guide(guide_name)

        elif subcommand == 'device':
            if len(args) < 2:
                print(f"Current device: {self.config.get('hardware.device')}")
                return
            device = args[1].lower()
            if device in ['auto', 'cpu', 'cuda', 'mps']:
                self.config.set('hardware.device', device)
                print(f"Device set to: {device}")
            else:
                print(f"Unsupported device: {device}")
                print("Supported devices: auto, cpu, cuda, mps")

        elif subcommand == 'batch':
            if len(args) < 2:
                print(f"Current batch settings:")
                print(f"  Exit on error: {self.config.get('batch.exit_on_error')}")
                print(f"  Timeout: {self.config.get('batch.timeout')} seconds")
                return
            batch_setting = args[1].lower()
            if batch_setting == 'exit_on_error':
                if len(args) < 3:
                    print(f"Current exit_on_error setting: {self.config.get('batch.exit_on_error')}")
                    return
                value = args[2].lower() in ['true', 'yes', '1', 't', 'y']
                self.config.set('batch.exit_on_error', value)
                print(f"Batch exit_on_error set to: {value}")
            elif batch_setting == 'timeout':
                if len(args) < 3:
                    print(f"Current timeout setting: {self.config.get('batch.timeout')} seconds")
                    return
                try:
                    timeout = int(args[2])
                    self.config.set('batch.timeout', timeout)
                    print(f"Batch timeout set to: {timeout} seconds")
                except ValueError:
                    print(f"Invalid timeout value: {args[2]}")
            else:
                print(f"Unknown batch setting: {batch_setting}")
                print("Available settings: exit_on_error, timeout")

        elif subcommand == 'hpc':
            if len(args) < 2:
                print(f"Current HPC mode: {self.config.get('system.hpc_mode')}")
                return
            hpc_mode = args[1].lower() in ['true', 'yes', '1', 't', 'y', 'on']
            self.config.set('system.hpc_mode', hpc_mode)
            # Update the instance variable too
            self.hpc_mode = hpc_mode
            print(f"HPC mode set to: {hpc_mode}")

        elif subcommand == 'export':
            # Export config to environment variables
            self.config.export_to_env()
            print("Configuration exported to environment variables")

        elif subcommand == 'reset':
            # Reset to defaults
            if len(args) < 2:
                print("Usage: /config reset <section.key> or 'all'")
                return

            target = args[1].lower()
            if target == 'all':
                # Re-initialize with defaults
                self.config._ensure_defaults()
                print("All configuration reset to defaults")
            else:
                # Try to reset specific section.key
                sections = target.split('.')
                if len(sections) != 2:
                    print("Invalid format. Use 'section.key' (e.g., 'system.debug_level')")
                    return

                section, key = sections
                if section not in self.config.config:
                    print(f"Section '{section}' not found in configuration")
                    return

                if key not in self.config.config[section]:
                    print(f"Key '{key}' not found in section '{section}'")
                    return

                # Reset using defaults
                self.config._ensure_defaults()
                default_value = self.config.config[section][key]

                # Set it back to trigger proper update behavior
                self.config.set(target, default_value)
                print(f"Reset {target} to default: {default_value}")

        else:
            print(f"Unknown config subcommand: {subcommand}")
            print(
                "Available subcommands: llm, embed, storage, kval, debug, output, guide, device, batch, hpc, export, reset")