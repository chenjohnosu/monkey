"""
Command line interface for the document analysis toolkit
"""

import os
import sys
import shlex
import datetime
from pathlib import Path
from core.engine.logging import debug, error, warning, info, trace
from core.engine.storage import StorageManager, VectorStoreInspector
from core.engine.output import OutputManager
from core.engine.interpreter import AnalysisInterpreter
from core.engine.utils import ensure_dir, format_size


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

        # Initialize the analysis interpreter
        self.analysis_interpreter = AnalysisInterpreter(
            config, self.storage_manager, self.output_manager, self.text_processor
        )

        self.saving_session = False
        self.loaded_workspaces = [self.current_workspace]
        self.active_guide = None

        # Create the batch processor
        from core.engine.batch import BatchProcessor
        self.batch_processor = BatchProcessor(self)

        info("CommandProcessor initialized")

    def run_interactive(self):
        """Run in interactive command mode"""
        debug(self.config, "Starting interactive command processing loop")
        info(f"Monkey v{self.config.get_version()} initialized. Type /help for available commands.")

        try:
            while self.running:
                # Simplified prompt showing only essential information
                prompt = f"{self.current_workspace}> "

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
        """Process commands from a batch file"""
        return self.batch_processor.process_file(batch_file)

    def process_command(self, command_string):
        """Process a command string"""
        debug(self.config, f"Processing command: {command_string}")

        # Echo the command to the output manager in non-HPC mode
        if not self.hpc_mode:
            print(f"> {command_string}")

        # Skip empty commands
        if not command_string.strip():
            return

        # Parse the command
        try:
            tokens = shlex.split(command_string)
            command = tokens[0].lower()
            args = tokens[1:]
        except Exception as e:
            print(f"Error parsing command: {str(e)}")
            return

        # Handle commands
        if command.startswith('/'):
            # Process system command
            self._handle_system_command(command[1:], args)
        else:
            # Treat as a query in interactive mode
            if hasattr(self, 'query_engine') and self.query_engine.is_active():
                self.query_engine.process_query(command_string)
            else:
                print("Not in query mode. Use '/run query' to enter interactive query mode.")

    def _handle_system_command(self, command, args):
        """Handle system commands"""
        debug(self.config, f"Handling system command: {command} with args: {args}")

        # Handle aliases
        command = self._resolve_alias(command)

        # Special handling for exit/quit in query mode
        if (command == 'exit' or command == 'quit') and hasattr(self, 'query_engine') and self.query_engine.is_active():
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

    def _cmd_quit(self):
        """Handle the quit command"""
        debug(self.config, "Quit command received")
        self.running = False

    def _cmd_help(self, args):
        """Display help information"""
        debug(self.config, f"Help command with args: {args}")

        if not args:
            self._display_help_overview()
        elif args[0] in ['run', 'load', 'config', 'save', 'show', 'inspect', 'explain', 'clear']:
            self._show_specific_help(args[0])
        else:
            print(f"No specific help available for '{args[0]}'")

    def _display_help_overview(self):
        """Display general help information"""
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

  Operations:
    /load               - Load workspace
    /save               - Start/stop saving session or buffer
    /config             - Set runtime configuration
    /show               - Show active information and data sources
    /clear              - Clear logs, vector database, or cache files
    /inspect            - Check/verify status of data stores 
    /quit, /exit        - Exit the application
    /help [command]     - Display help information

  Batch Processing:
    Use -b/--batch flag with a batch file path
    Use --hpc flag for high-performance computing optimization
    Use --cmd flag to run a single command
    """)

    def _show_specific_help(self, topic):
        """Show help for a specific topic"""
        if topic == 'run':
            print("""
Run Commands:
  /run themes [method]     - Run theme analysis
    Methods: all, nfm, net, key, lsa, cluster
      all    - Run all theme analysis methods
      nfm    - Named entity-based theme extraction
      net    - Content network analysis
      key    - Keyword-based theme identification
      lsa    - Latent semantic analysis
      cluster - Document clustering

  /run query [query_text]  - Enter interactive query mode or run a single query

  /run grind               - Process files in workspace to create initial database

  /run update              - Update workspace with new or modified files

  /run scan [detailed]     - Scan workspace for new or updated files
    detailed - Show additional file information

  /run merge <source_ws>   - Merge source workspace into current workspace

  /run sentiment [method]  - Run sentiment analysis
    Methods: all, basic, advanced
      all     - Run all sentiment analysis methods
      basic   - Simple sentiment classification
      advanced - In-depth analysis with aspect extraction

  /run topics [method]     - Run topic modeling
    Methods: all, lda, nmf, cluster
      all    - Run all topic modeling methods
      lda    - Latent Dirichlet Allocation
      nmf    - Non-Negative Matrix Factorization
      cluster - Clustering-based topic modeling

Examples:
  /run themes nfm          - Run named entity-based theme extraction
  /run sentiment advanced  - Run advanced sentiment analysis
  /run topics lda          - Run LDA topic modeling
  /run query "What themes relate to climate?"  - Run a single query
            """)
        elif topic == 'load':
            print("""
Load Commands:
  /load ws <workspace>  - Load or create a workspace

Notes:
  - If the workspace doesn't exist, it will be created
  - Loading a workspace sets it as the current active workspace
  - All commands will operate on the current workspace
            """)
        elif topic == 'config':
            print("""
Config Commands:
  /config llm <model>                - Set LLM model (e.g., mistral, llama2)
  /config embed <model>              - Set embedding model (e.g., multilingual-e5, mixbread)
  /config storage <backend>          - Set storage backend (llama_index, haystack, or chroma)
  /config kval <n>                   - Set k value for retrieval (number of docs to return)
  /config debug [on|off]             - Enable or disable debug mode
  /config output [txt|json]          - Set output format for saved files
  /config guide <guide>              - Set guide from guides.txt

Examples:
  /config llm mistral                - Set LLM model to mistral
  /config embed multilingual-e5      - Set embedding model to multilingual-e5
  /config kval 5                     - Set retrieval to return 5 documents
            """)
        elif topic == 'save':
            print("""
Save Commands:
  /save start                - Start saving session (all commands and outputs)
  /save stop                 - Stop saving session and write to file
  /save buffer               - Save last command output to file

Notes:
  - Session logs are saved to logs/<workspace>/ directory
  - Output format depends on your config.output_format setting
  - The buffer contains the most recent command output
            """)
        elif topic == 'show':
            print("""
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
            """)
        elif topic == 'inspect':
            print("""
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
            """)
        elif topic == 'explain':
            print("""
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
            """)
        elif topic == 'clear':
            print("""
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
            """)
        else:
            print(f"No specific help available for '{topic}'")

    def _cmd_run(self, args):
        """Handle the run command"""
        debug(self.config, f"Run command with args: {args}")

        if not args:
            print("Usage: /run [themes|query|grind|update|scan|merge|sentiment|topics] [options]")
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
                if not self.batch_mode and not self.hpc_mode:
                    print("\nRESPONSE:")
                    print(response)
                    print()
            else:
                # Enter interactive query mode
                if self.query_engine.activate(self.current_workspace):
                    if not self.batch_mode and not self.hpc_mode:
                        print(f"Entering interactive query mode for workspace '{self.current_workspace}'")
                        print("Type your queries directly. Use /exit to return to command mode.")
                    # In batch mode, just keep query mode active for subsequent commands

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
            print("Available subcommands: themes, query, grind, update, scan, merge, sentiment, topics")

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
        """Load a workspace with confirmation for new workspaces"""
        debug(self.config, f"Loading workspace: {workspace}")

        # Check if workspace directories exist
        from pathlib import Path
        data_dir = Path("data") / workspace
        body_dir = Path("body") / workspace

        # Check if this is a new workspace
        is_new_workspace = not data_dir.exists() and not body_dir.exists()

        if is_new_workspace:
            # Ask for confirmation before creating a new workspace
            # Skip confirmation in HPC/batch mode
            if not self.hpc_mode and not self.batch_mode:
                confirm = input(f"Workspace '{workspace}' does not exist. Create it? (y/n): ").strip().lower()
                if confirm != 'y' and confirm != 'yes':
                    print(f"Cancelled creation of workspace '{workspace}'")
                    return
            else:
                # In HPC/batch mode, automatically create workspace
                info(f"Creating new workspace: {workspace}")

        # Create necessary directories
        for directory in [data_dir, body_dir]:
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                debug(self.config, f"Created directory: {directory}")

        # Add to loaded workspaces if not already loaded
        if workspace not in self.loaded_workspaces:
            self.loaded_workspaces.append(workspace)
            debug(self.config, f"Workspace '{workspace}' loaded")

        # Update current workspace
        if self.current_workspace != workspace:
            self.current_workspace = workspace
            self.config.set('workspace.default', workspace)
            if not self.hpc_mode:
                print(f"Switched active workspace to '{workspace}'")

    def _cmd_save(self, args):
        """Handle the save command"""
        debug(self.config, f"Save command with args: {args}")

        if not args:
            print("Usage: /save [start|stop|buffer]")
            return

        subcommand = args[0].lower()

        if subcommand == 'start':
            self._start_saving_session()
        elif subcommand == 'stop':
            self._stop_saving_session()
        elif subcommand == 'buffer':
            self._save_buffer()
        else:
            print(f"Unknown save subcommand: {subcommand}")
            print("Available subcommands: start, stop, buffer")

    def _start_saving_session(self):
        """Start saving session output"""
        debug(self.config, "Starting session saving")

        if self.saving_session:
            print("Session saving is already active")
            return

        self.saving_session = True
        self.output_manager.start_session_saving(self.current_workspace)
        if not self.batch_mode and not self.hpc_mode:
            print("Session saving started")

    def _stop_saving_session(self):
        """Stop saving session output"""
        debug(self.config, "Stopping session saving")

        if not self.saving_session:
            print("Session saving is not active")
            return

        self.saving_session = False
        filepath = self.output_manager.stop_session_saving()
        if filepath and not self.hpc_mode:
            print(f"Session saving stopped. Output saved to: {filepath}")

    def _save_buffer(self):
        """Save last command output buffer"""
        debug(self.config, "Saving output buffer")

        filepath = self.output_manager.save_buffer(self.current_workspace)
        if filepath:
            if not self.hpc_mode:
                print(f"Buffer saved to: {filepath}")
        else:
            print("No buffer to save")

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

        else:
            print(f"Unknown config subcommand: {subcommand}")
            print("Available subcommands: llm, embed, storage, kval, debug, output, guide")

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
                if not self.hpc_mode:
                    print(f"Guide '{guide_name}' loaded.")

                # Store the guide for later use
                self.config.set_guide(guide_name, guide_content)

                # Set as active guide
                self.active_guide = guide_name
                if not self.hpc_mode:
                    print(f"Guide '{guide_name}' set as active guide")
            else:
                print(f"Guide '{guide_name}' not found in guides.txt")
        except FileNotFoundError:
            print("guides.txt file not found")
        except Exception as e:
            print(f"Error loading guide: {str(e)}")

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
        else:
            print(f"Unknown show subcommand: {subcommand}")
            print("Available subcommands: status, cuda, config, ws, files, guide")

    def _show_status(self):
        """Show system status"""
        debug(self.config, "Showing system status")

        print("\nSYSTEM STATUS")
        print(f"Current Workspace: {self.current_workspace}")
        print(f"LLM Model: {self.config.get('llm.default_model')}")
        print(f"Embedding Model: {self.config.get('embedding.default_model')}")
        print(f"Debug Mode: {self.config.get('system.debug_level')}")
        print(f"Output Format: {self.config.get('system.output_format')}")
        print(f"Device: {self.config.get('hardware.device')}")
        print(f"Session Saving: {'Active' if self.saving_session else 'Inactive'}")

    def _show_cuda(self):
        """Show CUDA status"""
        debug(self.config, "Showing CUDA status")

        from core.engine.cuda import check_cuda_status
        check_cuda_status(self.config)

    def _show_config(self):
        """Show configuration details"""
        debug(self.config, "Showing configuration")
        self.config.display()

    def _show_workspace(self):
        """Show workspace details"""
        debug(self.config, "Showing workspace details")

        print("\nWORKSPACES")
        for ws in self.loaded_workspaces:
            if ws == self.current_workspace:
                print(f"  {ws} (active)")
            else:
                print(f"  {ws}")

        # Get additional workspace stats if available
        if hasattr(self, 'storage_manager'):
            stats = self.storage_manager.get_workspace_stats(self.current_workspace)
            if stats:
                print(f"\nDetails for {self.current_workspace}")
                print(f"  Document Count: {stats.get('doc_count', 0)}")
                print(f"  Embeddings: {stats.get('embedding_count', 0)}")
                print(f"  Last Updated: {stats.get('last_updated', 'Unknown')}")
                print(f"  Embedding Model: {stats.get('embedding_model', 'Unknown')}")

                # Show language distribution
                languages = stats.get('languages', {})
                if languages:
                    print("\nLanguage Distribution")
                    for lang, count in languages.items():
                        print(f"    {lang}: {count}")

    def _show_files(self):
        """Show files in the current workspace"""
        debug(self.config, "Showing workspace files")

        import os
        import datetime

        doc_dir = os.path.join("body", self.current_workspace)

        if not os.path.exists(doc_dir):
            print(f"Document directory does not exist: {doc_dir}")
            return

        print(f"\nFILES IN WORKSPACE: {self.current_workspace}")

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
            print("No files found")
        else:
            # Print file list
            print(f"\n  Filename  Size  Modified")

            for file_path, size, mod_time in sorted(files):
                print(f"  {file_path}  {format_size(size)}  {mod_time}")

                def _show_guides(self):
                    """Show available guides"""
                    debug(self.config, "Showing available guides")

                    try:
                        with open("guides.txt", "r", encoding="utf-8") as file:
                            content = file.read()

                        import re
                        # Find all guide tags
                        guide_tags = re.findall(r'<([^>]+)>.*?</\1>', content, re.DOTALL)

                        if not guide_tags:
                            print("No guides found in guides.txt")
                            return

                        print("Available Guides:")
                        for guide in sorted(guide_tags):
                            if guide == self.active_guide:
                                print(f"  * {guide} (active)")
                            else:
                                print(f"    {guide}")

                        print("\nUse '/config guide <name>' to set a guide as active")

                    except FileNotFoundError:
                        print("guides.txt file not found")
                    except Exception as e:
                        print(f"Error reading guides: {str(e)}")

                def _cmd_inspect(self, args):
                    """Handle the inspect command"""
                    debug(self.config, f"Inspect command with args: {args}")

                    if not args:
                        print(
                            "Usage: /inspect [workspace|ws|documents|vectorstore|vdb|query|rebuild|fix|metadata|migrate]")
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
                        self._inspect_documents(workspace, limit)
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
                        self._inspect_query(workspace, query)
                    elif subcommand == 'rebuild':
                        workspace = args[1] if len(args) > 1 else self.current_workspace
                        self._rebuild_vector_store(workspace)
                    elif subcommand == 'fix':
                        workspace = args[1] if len(args) > 1 else self.current_workspace
                        print(f"Attempting to fix vector store issues for workspace '{workspace}'...")
                        self._fix_vector_store(workspace)
                    elif subcommand == 'metadata':
                        workspace = args[1] if len(args) > 1 else self.current_workspace
                        test_query = args[2] if len(args) > 2 else "test"
                        limit = int(args[3]) if len(args) > 3 and args[3].isdigit() else 3
                        self._inspect_metadata(workspace, test_query, limit)
                    elif subcommand == 'migrate':
                        workspace = args[1] if len(args) > 1 else self.current_workspace
                        print(f"Migrating vector store for workspace '{workspace}'...")
                        success = self._migrate_vector_store(workspace)
                        if success:
                            print("Migration completed successfully.")
                        else:
                            print("Migration not needed or failed. Check logs for details.")
                    else:
                        print(f"Unknown inspect subcommand: {subcommand}")
                        print(
                            "Available subcommands: workspace, documents, vectorstore, query, rebuild, fix, metadata, migrate")

                def _inspect_documents(self, workspace, limit=5):
                    """Dump document content for inspection"""
                    debug(self.config, f"Inspecting documents in workspace: {workspace}")

                    # Get documents
                    docs = self.storage_manager.get_documents(workspace)
                    if not docs:
                        print(f"No documents found in workspace '{workspace}'")
                        return

                    print(f"\nDocument Content in Workspace: {workspace}")
                    print(f"Total documents: {len(docs)} (showing first {min(limit, len(docs))})")

                    for i, doc in enumerate(docs[:limit]):
                        source = doc.get('metadata', {}).get('source', 'unknown')
                        language = doc.get('metadata', {}).get('language', 'unknown')

                        print(f"\n--- Document {i + 1}: {source} ---")
                        print(f"Language: {language}")

                        content = doc.get('content', '')
                        if len(content) > 1000:
                            content = content[:1000] + "... [truncated]"

                        print("\nContent:")
                        print(content)

                        print("\nMetadata:")
                        for key, value in doc.get('metadata', {}).items():
                            print(f"  {key}: {value}")

                    if len(docs) > limit:
                        print(f"\n... and {len(docs) - limit} more documents")

                def _inspect_query(self, workspace, query="test"):
                    """Test query pipeline with a sample query"""
                    debug(self.config, f"Testing query pipeline for workspace: {workspace}")

                    print(f"\nTesting Query Pipeline for Workspace: {workspace}")
                    print(f"Test Query: '{query}'")

                    # Activate query engine
                    if not self.query_engine.is_active() or self.query_engine.current_workspace != workspace:
                        print(f"Activating query engine for workspace '{workspace}'...")
                        if not self.query_engine.activate(workspace):
                            print(f"Failed to activate query engine for workspace '{workspace}'")
                            return

                    # Process the test query
                    try:
                        print("Processing query...")
                        self.query_engine.process_query(query)
                        print("Query processed successfully.")
                    except Exception as e:
                        print(f"Error processing query: {str(e)}")
                    finally:
                        # Deactivate query engine unless we're in batch mode
                        if not self.batch_mode:
                            self.query_engine.deactivate()

                def _rebuild_vector_store(self, workspace):
                    """Rebuild vector store from documents"""
                    debug(self.config, f"Rebuilding vector store for workspace: {workspace}")

                    print(f"\nRebuilding Vector Store for Workspace: {workspace}")

                    # Confirm action
                    if not self.hpc_mode and not self.batch_mode:
                        confirm = input(
                            f"Are you sure you want to rebuild the vector store for workspace '{workspace}'? (y/n): ").strip().lower()
                        if confirm != 'y' and confirm != 'yes':
                            print("Rebuild cancelled.")
                            return

                    # Get existing documents
                    docs = self.storage_manager.get_documents(workspace)
                    if not docs:
                        print(f"No documents found in workspace '{workspace}'")
                        return

                    print(f"Found {len(docs)} documents for rebuilding vector store")

                    # Backup existing vector store
                    vector_dir = os.path.join("data", workspace, "vector_store")
                    if os.path.exists(vector_dir):
                        from core.engine.utils import create_timestamped_backup
                        backup_dir = create_timestamped_backup(vector_dir)
                        if backup_dir:
                            print(f"Created backup of existing vector store: {backup_dir}")

                        # Remove existing vector store
                        import shutil
                        shutil.rmtree(vector_dir)
                        os.makedirs(vector_dir)
                        print("Removed existing vector store")

                    # Create new vector store
                    print("Creating new vector store...")
                    success = self.storage_manager.create_vector_store(workspace)

                    if success:
                        print("Vector store rebuilt successfully")
                    else:
                        print("Failed to rebuild vector store")

                def _fix_vector_store(self, workspace):
                    """Fix common vector store issues"""
                    debug(self.config, f"Fixing vector store issues for workspace: {workspace}")

                    print(f"\nAttempting to fix vector store issues for Workspace: {workspace}")

                    # Step 1: Check if vector store directory exists
                    vector_dir = os.path.join("data", workspace, "vector_store")
                    if not os.path.exists(vector_dir):
                        print(f"Vector store directory does not exist: {vector_dir}")
                        print("Creating empty vector store...")
                        self.storage_manager.load_vector_store(workspace)
                        return

                    # Step 2: Check for common inconsistencies
                    vector_store_json = os.path.join(vector_dir, "vector_store.json")
                    inconsistent_names = [
                        f for f in os.listdir(vector_dir)
                        if f.endswith('_vector_store.json') or f.endswith('__vector_store.json')
                    ]

                    if inconsistent_names and not os.path.exists(vector_store_json):
                        print(f"Found inconsistent vector store file names: {inconsistent_names}")

                        # Find largest file as the main one
                        largest_file = max(inconsistent_names,
                                           key=lambda f: os.path.getsize(os.path.join(vector_dir, f)))
                        print(f"Using {largest_file} as the main vector store file")

                        # Copy to correct name
                        import shutil
                        shutil.copyfile(
                            os.path.join(vector_dir, largest_file),
                            vector_store_json
                        )
                        print(f"Fixed vector store filename inconsistency")

                    # Step 3: Verify document store
                    docstore_path = os.path.join(vector_dir, "docstore.json")
                    if not os.path.exists(docstore_path):
                        print(f"Document store file missing: {docstore_path}")
                        print("Rebuilding vector store...")
                        self._rebuild_vector_store(workspace)
                        return

                    # Step 4: Try loading the vector store
                    try:
                        print("Testing vector store loading...")
                        loaded = self.storage_manager.load_vector_store(workspace)
                        if loaded:
                            print("Vector store loaded successfully. No major issues detected.")
                        else:
                            print("Failed to load vector store. Rebuilding recommended.")
                            if not self.hpc_mode and not self.batch_mode:
                                rebuild = input("Rebuild vector store now? (y/n): ").strip().lower()
                                if rebuild == 'y' or rebuild == 'yes':
                                    self._rebuild_vector_store(workspace)
                            else:
                                # In HPC/batch mode, automatically rebuild
                                self._rebuild_vector_store(workspace)
                    except Exception as e:
                        print(f"Error loading vector store: {str(e)}")
                        print("Vector store appears to be corrupted. Rebuilding recommended.")
                        if not self.hpc_mode and not self.batch_mode:
                            rebuild = input("Rebuild vector store now? (y/n): ").strip().lower()
                            if rebuild == 'y' or rebuild == 'yes':
                                self._rebuild_vector_store(workspace)
                        else:
                            # In HPC/batch mode, automatically rebuild
                            self._rebuild_vector_store(workspace)

                def _inspect_metadata(self, workspace, query="test", limit=3):
                    """Inspect raw metadata returned from vector store"""
                    debug(self.config, f"Inspecting vector store metadata for workspace: {workspace}")

                    print(f"\nInspecting vector store metadata for workspace: {workspace}")
                    debug(f"Test query: '{query}'")

                    # Load vector store
                    loaded = self.storage_manager.load_vector_store(workspace)
                    if not loaded:
                        print("Failed to load vector store")
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
                        print(f"Error: {str(e)}")

                def _migrate_vector_store(self, workspace):
                    """Migrate vector store to handle naming changes"""
                    debug(self.config, f"Migrating vector store for workspace: {workspace}")

                    vector_dir = os.path.join("data", workspace, "vector_store")
                    if not os.path.exists(vector_dir):
                        print(f"Vector store directory does not exist: {vector_dir}")
                        return False

                    # Check for naming inconsistencies
                    vector_store_json = os.path.join(vector_dir, "vector_store.json")
                    inconsistent_names = [
                        f for f in os.listdir(vector_dir)
                        if f.endswith('_vector_store.json') or f.endswith('__vector_store.json')
                    ]

                    if inconsistent_names and not os.path.exists(vector_store_json):
                        print(f"Found inconsistent vector store file names: {inconsistent_names}")

                        # Find largest file as the main one
                        largest_file = max(inconsistent_names,
                                           key=lambda f: os.path.getsize(os.path.join(vector_dir, f)))
                        print(f"Using {largest_file} as the main vector store file")

                        # Copy to correct name
                        import shutil
                        shutil.copyfile(
                            os.path.join(vector_dir, largest_file),
                            vector_store_json
                        )
                        print(f"Migrated vector store naming to standard format")
                        return True

                    print("No migration needed for this vector store")
                    return False

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
                        print(f"Invalid subcommand: {subcommand}")
                        print(f"Must be one of: themes, topics, sentiment, session")
                        return

                    # Normalize the subcommand to the standard form
                    normalized_subcommand = valid_subcommands[subcommand]

                    # Get the question (everything after the subcommand)
                    question = ' '.join(args[1:]) if len(args) > 1 else None

                    # Run the appropriate interpretation based on subcommand
                    try:
                        workspace = self.current_workspace
                        if not self.hpc_mode:
                            print(f"Analyzing {normalized_subcommand} results, please wait...")

                        interpretation = self.analysis_interpreter.interpret_analysis(
                            workspace, normalized_subcommand, query=question
                        )

                        # Display interpretation with proper formatting
                        if not self.hpc_mode:
                            print(f"\nLLM EXPLANATION OF {normalized_subcommand.upper()} ANALYSIS")
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

                        if not self.hpc_mode:
                            print(f"Explanation saved to: {filepath}")

                    except Exception as e:
                        print(f"Error generating explanation: {str(e)}")
                        import traceback
                        traceback.print_exc()

                def _cmd_clear(self, args):
                    """Handle the clear command to remove logs, vector databases, or cached data"""
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

                    # Ask for confirmation unless in HPC mode
                    if not self.hpc_mode and not self.batch_mode:
                        confirm = input(
                            f"Are you sure you want to clear all logs for workspace '{workspace}'? (y/n): ").strip().lower()
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

                    # Ask for confirmation unless in HPC mode
                    if not self.hpc_mode and not self.batch_mode:
                        confirm = input(
                            f"Are you sure you want to clear the vector database for workspace '{workspace}'? This will require rebuilding indexes. (y/n): ").strip().lower()
                        if confirm != 'y' and confirm != 'yes':
                            print("Operation cancelled")
                            return

                    # Create backup
                    from core.engine.utils import create_timestamped_backup
                    backup_dir = create_timestamped_backup(vector_dir)
                    if backup_dir:
                        print(f"Created backup of vector database at: {backup_dir}")

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

                    # Ask for confirmation unless in HPC mode
                    if not self.hpc_mode and not self.batch_mode:
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