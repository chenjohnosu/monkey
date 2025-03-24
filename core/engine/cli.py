"""
Command line interface for the document analysis toolkit
"""

import shlex
from core.engine.logging import debug_print
from core.engine.storage import StorageManager, VectorStoreInspector
from core.engine.output import OutputManager
from core.engine.interpreter import setup_llm_theme_analysis
from datetime import datetime

class CommandProcessor:
    """Command line interface processor"""

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

        # Initialize LLM theme integration
        self._init_llm_theme_integration()

        debug_print(config, "CommandProcessor initialized")

    def start(self):
        """Start the command processing loop"""
        debug_print(self.config, "Starting command processing loop")
        print(f"Monkey v{self.config.get_version()} initialized. Type /help for available commands.")

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
        """Handle system commands"""
        debug_print(self.config, f"Handling system command: {command} with args: {args}")

        # Handle aliases
        command = self._resolve_alias(command)

        # Special handling for exit/quit in query mode
        if (command == 'exit' or command == 'quit') and hasattr(self, 'query_engine') and self.query_engine.is_active():
            # Just exit query mode, not the whole application
            debug_print(self.config, "Exit/quit command in query mode - deactivating query mode")
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
            'i': 'inspect'
        }
        return aliases.get(command, command)

    def _cmd_quit(self):
        """Handle the quit command"""
        debug_print(self.config, "Quit command received")
        self.running = False

    def _cmd_help(self, args):
        """Display help information"""
        debug_print(self.config, f"Help command with args: {args}")

        if not args:
            print("""
    Available Commands:
      System:
        /quit, /exit              - Exit the application
        /help [topic]             - Display help information
        /show [status|cuda]       - Show system information
        /show [guide|config]      - Show configurable information
        /show [ws|files]          - Show data sources

      Run Modes:
        /run themes [all|nfm|net|key|lsa|cluster] - Run theme analysis
        /run themes-llm [all|nfm|net|key|lsa|cluster] - Run theme analysis with LLM interpretation
        /run query                 - Enter interactive query mode
        /run grind             - Process files in workspace to create initial database
        /run update            - Update workspace with new or modified files
        /run scan              - Scan workspace for new or updated files
        /run merge       - Merge workspaces
        /run sentiment [all|basic|advanced] - Run sentiment analysis
        /run topic [all|lda|nmf|cluster]    - Run topic modeling

      File Operations:
        /load <workspace>          - Load workspace

      Output:
        /save start                - Start saving session
        /save stop                 - Stop saving session
        /save buffer               - Save last command output

      Configuration:
        /config llm <model>        - Set LLM model
        /config embed <model>      - Set embedding model
        /config storage <backend>  - Set storage backend (llama_index, haystack, or chroma)
        /config kval <n>           - Set k value for retrieval
        /config debug [on|off]     - Set debug mode
        /config output [txt|md|json] - Set output format
        /config guide <guide>      - Set guide from guides.txt

      Inspection:
        /inspect workspace [ws]    - Inspect workspace metadata and files
        /inspect documents [ws]    - Dump document content
        /inspect vectorstore [ws]  - Dump vector store metadata
        /inspect query [ws] [q]    - Test query pipeline
        /inspect rebuild [ws]      - Rebuild vector store

    Aliases:
      /q - /quit, /c - /config, /l - /load, /r - /run, /s - /save, /h - /help, /i - /inspect
    """)
        elif args[0] in ['run', 'load', 'config', 'save', 'show', 'inspect']:
            self._show_specific_help(args[0])
        else:
            print(f"No specific help available for '{args[0]}'")

    def _show_specific_help(self, topic):
        """Show help for a specific topic"""
        if topic == 'run':
            print("""
    Run Commands:
      /run themes [all|nfm|net|key|lsa|cluster]  - Run theme analysis
      /run themes-llm [all|nfm|net|key|lsa|cluster] - Run theme analysis with LLM interpretation
      /run query                                 - Enter interactive query mode
      /run grind                                 - Process files in workspace to create initial database
      /run update                                - Update workspace with new or modified files
      /run scan [detailed]                       - Scan workspace for new or updated files
      /run merge <source_workspace>              - Merge source workspace into current workspace
      /run sentiment [all|basic|advanced]        - Run sentiment analysis
      /run topic [all|lda|nmf|cluster]           - Run topic modeling
    """)
        elif topic == 'load':
            print("""
    Load Commands:
      /load <workspace>  - Load or create a workspace
    """)
        elif topic == 'config':
            print("""
    Config Commands:
      /config llm <model>                - Set LLM model
      /config embed <model>              - Set embedding model
      /config storage <backend>          - Set storage backend (llama_index, haystack, or chroma)
      /config kval <n>                   - Set k value for retrieval
      /config debug [on|off]             - Set debug mode
      /config output [txt|md|json]       - Set output format
      /config guide <guide>              - Set guide from guides.txt
    """)
        elif topic == 'save':
            print("""
    Save Commands:
      /save start                - Start saving session
      /save stop                 - Stop saving session
      /save buffer               - Save last command output
    """)
        elif topic == 'show':
            print("""
    Show Commands:
      /show status               - Show system status
      /show cuda                 - Show CUDA status
      /show config               - Show configuration details
      /show ws                   - Show workspace details
      /show files                - Show files in current workspace
      /show guide                - Show available guides
    """)
        elif topic == 'inspect':
            print("""
    Inspect Commands:
      /inspect workspace [ws] - Inspect workspace metadata and vector store
      /inspect ws [ws]        - Alias for workspace inspect
      /inspect documents [ws] [limit] - Dump document content in workspace
        Optional: Specify number of documents to show (default: 5)
      /inspect vectorstore [ws] - Dump vector store metadata
      /inspect vdb [ws]         - Alias for vectorstore inspect
      /inspect query [ws] [query]     - Test query pipeline with optional test query
      /inspect rebuild [ws]           - Rebuild vector store from existing documents
      /inspect fix [ws]               - Fix common vector store issues
      /inspect metadata [ws] [query]  - Inspect raw metadata returned from vector store
      /inspect migrate [ws]           - Fix inconsistent vector store naming
    """)
        else:
            print(f"No specific help available for '{topic}'")

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

    def _format_size(self, size_bytes):
        """Format file size in a human-readable format"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024 or unit == 'GB':
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024

    def _cmd_run(self, args):
        """Handle the run command"""
        debug_print(self.config, f"Run command with args: {args}")

        if not args:
            print("Usage: /run [themes|themes-llm|query|grind|update|scan|merge|sentiment|topic] [options]")
            return

        subcommand = args[0].lower()

        if subcommand == 'themes':
            method = args[1] if len(args) > 1 else 'all'
            self.theme_analyzer.analyze(self.current_workspace, method)

        elif subcommand == 'themes-llm':
            # Import the module here to avoid circular imports
            from core.engine.interpreter import run_themes_with_llm_interpretation

            # Extract method parameter if provided
            method = args[1] if len(args) > 1 else 'all'

            # Create a method reference bound to self
            import types
            run_method = types.MethodType(run_themes_with_llm_interpretation, self)

            # Call the method
            run_method(self.current_workspace, method)

        elif subcommand == 'query':
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

        elif subcommand == 'topic':
            method = args[1] if len(args) > 1 else 'all'
            self.topic_modeler.analyze(self.current_workspace, method)

        else:
            print(f"Unknown run subcommand: {subcommand}")

    def _init_llm_theme_integration(self):
        """Initialize LLM-assisted theme analysis capability"""
        try:
            # Import the setup function
            from core.engine.interpreter import setup_llm_theme_analysis

            # Call the setup function with self as the command processor
            setup_llm_theme_analysis(self)

            # Report success
            debug_print(self.config, "LLM theme analysis integration initialized")
        except Exception as e:
            debug_print(self.config, f"Error initializing LLM theme analysis: {str(e)}")
            print("LLM-assisted theme analysis feature could not be initialized.")

    def _cmd_load(self, args):
        """Handle the load command"""
        debug_print(self.config, f"Load command with args: {args}")

        if not args:
            print("Usage: /load <workspace>")
            return

        # Simplified load command - always for workspace
        workspace = args[0]
        self._load_workspace(workspace)

    def _show_guides(self):
        """Show available guides"""
        debug_print(self.config, "Showing available guides")

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

    def _load_workspace(self, workspace):
        """Load a workspace"""
        debug_print(self.config, f"Loading workspace: {workspace}")

        # Ensure workspace directories exist
        import os
        data_dir = os.path.join("data", workspace)
        body_dir = os.path.join("body", workspace)

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

    def _cmd_config(self, args):
        """Handle the config command"""
        debug_print(self.config, f"Config command with args: {args}")

        if not args:
            print("Usage: /config [llm|embed|kval|debug|storage|output] [value]")
            return

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
                return
            self.config.set('embedding.default_model', args[1])
            print(f"Embedding model set to: {args[1]}")
        elif subcommand == 'kval':
            if len(args) < 2:
                print(f"Current k value: {self.config.get('query.k_value')}")
                return
            try:
                k_value = int(args[1])
                self.config.set('query.k_value', k_value)
                print(f"K value set to: {k_value}")
            except ValueError:
                print("K value must be an integer")
        elif subcommand == 'debug':
            if len(args) < 2:
                print(f"Debug mode: {'Enabled' if self.config.get('system.debug') else 'Disabled'}")
                return
            debug_value = args[1].lower() == 'on'
            self.config.set('system.debug', debug_value)
            print(f"Debug mode: {'Enabled' if debug_value else 'Disabled'}")
        elif subcommand == 'storage':
            if len(args) < 2:
                print(f"Current storage backend: {self.config.get('storage.vector_store')}")
                return
            valid_backends = ['llama_index', 'haystack', 'chroma']
            if args[1] not in valid_backends:
                print(f"Storage backend must be one of: {', '.join(valid_backends)}")
                return
            self.config.set('storage.vector_store', args[1])
            print(f"Storage backend set to: {args[1]}")
        elif subcommand == 'output':
            if len(args) < 2:
                print(f"Current output format: {self.config.get('system.output_format')}")
                return
            if args[1] not in ['txt', 'md', 'json']:
                print("Output format must be one of: txt, md, json")
                return
            self.config.set('system.output_format', args[1])
            print(f"Output format set to: {args[1]}")
        elif subcommand == 'guide':
            if len(args) < 2:
                print(f"Current active guide: {self.active_guide or 'None'}")
                return
            self._load_guide(args[1])
        else:
            print(f"Unknown config subcommand: {subcommand}")

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
                    print(f"Error inspecting LlamaIndex store: {str(e)}")
        elif subcommand == 'query':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            query = args[2] if len(args) > 2 else "test"
            self.vector_store_inspector.test_query_pipeline(workspace, query)
        elif subcommand == 'rebuild':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            self.vector_store_inspector.rebuild_vector_store(workspace)
        elif subcommand == 'fix':
            workspace = args[1] if len(args) > 1 else self.current_workspace
            print(f"Attempting to fix vector store issues for workspace '{workspace}'...")
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
            print(f"Unknown inspect subcommand: {subcommand}")

    def _inspect_metadata(self, workspace, query="test", limit=3):
        """Inspect raw metadata returned from vector store"""
        print(f"\nInspecting vector store metadata for workspace: {workspace}")
        print(f"Test query: '{query}'")

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
        debug_print(self.config, "Showing workspace files")

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
                import datetime
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
        """Process a command string with compact colored output"""
        debug_print(self.config, f"Processing command: {command_string}")

        # Skip empty commands
        if not command_string.strip():
            return

        # Log the command nicely
        self.output_manager.print_formatted('command', command_string)

        # Parse the command (existing code)
        try:
            tokens = shlex.split(command_string)
            command = tokens[0].lower()
            args = tokens[1:]
        except Exception as e:
            self.output_manager.print_formatted('feedback', f"Error parsing command: {str(e)}", success=False)
            return

        # Handle commands (rest of existing code)
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

