import os
import yaml
from core.engine.logging import debug, error, info, warning

class Config:
    def __init__(self, config_path='config.yaml'):
        self.config_path = config_path
        self.loaded_guides = {}
        self.version = '0.9.0'
        self.env_vars_loaded = False

        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                if not self.config:
                    self.config = {}
        except FileNotFoundError:
            warning(f"Config file not found: {config_path}. Using default configuration.")
            self.config = {}
        except yaml.YAMLError as e:
            error(f"Error parsing config file: {str(e)}. Using default configuration.")
            self.config = {}

        # Load environment variables
        self._load_env_vars()

        # Initialize with defaults if needed
        self._ensure_defaults()

        # Update logging configuration
        from core.engine.logging import LogManager
        debug_level = self.get('system.debug_level', 'info')
        LogManager.set_level(debug_level)

        # Debug output
        if debug_level in ['debug', 'trace']:
            debug(f"Configuration loaded from {config_path}")

    def _load_env_vars(self):
        """Load configuration from environment variables"""
        if self.env_vars_loaded:
            return

        # Look for environment variables starting with MONKEY_
        for key, value in os.environ.items():
            if key.startswith('MONKEY_'):
                # Convert environment variable name to config path
                # e.g., MONKEY_SYSTEM_DEBUG_LEVEL -> system.debug_level
                config_path = key[7:].lower().replace('_', '.')

                # Convert value types
                if value.lower() in ['true', 'yes', '1']:
                    value = True
                elif value.lower() in ['false', 'no', '0']:
                    value = False
                elif value.isdigit():
                    value = int(value)
                elif value.replace('.', '', 1).isdigit() and value.count('.') == 1:
                    value = float(value)

                # Set the value
                debug(f"Loading from environment: {config_path}={value}")
                self.set(config_path, value)

        self.env_vars_loaded = True

    def _ensure_defaults(self):
        """Ensure default configuration values are set"""
        defaults = {
            'system': {
                'debug_level': 'off',
                'output_format': 'txt',
                'hpc_mode': False,
                'batch_mode': False,
                'auto_save': True  # Auto-save results in batch/HPC mode
            },
            'hardware': {
                'use_cuda': 'auto',
                'use_mps': 'auto',
                'device': 'auto',
                'threads': 0  # 0 means auto-detect
            },
            'llm': {
                'default_model': 'mistral',
                'ollama_host': 'http://localhost',
                'ollama_port': 11434
            },
            'embedding': {
                'default_model': 'multilingual-e5'
            },
            'query': {
                'k_value': 5
            },
            'language': {
                'supported': ['en', 'zh'],
                'default': 'en'
            },
            'workspace': {
                'default': 'default'
            },
            'storage': {
                'vector_store': 'llama_index'
            },
            'topic': {
                'use_originals': True
            },
            'batch': {
                'exit_on_error': True,
                'timeout': 3600  # 1 hour max execution time
            }
        }

        # Ensure all default settings exist
        for section, values in defaults.items():
            if section not in self.config:
                self.config[section] = {}

            for key, value in values.items():
                if key not in self.config[section]:
                    self.config[section][key] = value

    def get(self, path, default=None):
        """Get a configuration value by path (e.g., 'system.debug_level')"""
        sections = path.split('.')
        config = self.config

        for section in sections:
            if section not in config:
                return default
            config = config[section]

        return config

    def set(self, path, value):
        """Set a configuration value by path (e.g., 'system.debug_level')"""
        sections = path.split('.')
        config = self.config

        # Navigate to the last section
        for section in sections[:-1]:
            if section not in config:
                config[section] = {}
            config = config[section]

        # Set the value
        config[sections[-1]] = value

        # Save the updated configuration
        if self.get('system.auto_save', True) and not self.get('system.hpc_mode', False):
            self._save_config()

        # Debug output
        debug(f"Configuration updated: {path} = {value}")

    def _save_config(self):
        """Save the configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        except Exception as e:
            error(f"Error saving configuration: {str(e)}")

    def set_guide(self, name, content):
        """Store a loaded guide"""
        self.loaded_guides[name] = content

    def get_guide(self, name):
        """Get a loaded guide"""
        return self.loaded_guides.get(name)

    def get_version(self):
        """Get the application version"""
        return self.version

    def display(self):
        """Display the current configuration"""
        print("Current Configuration:")

        for section, values in self.config.items():
            print(f"[{section}]")
            for key, value in values.items():
                print(f"  {key}: {value}")
            print()

    def export_to_env(self):
        """Export configuration to environment variables"""
        for section, values in self.config.items():
            for key, value in values.items():
                # Convert config path to environment variable name
                # e.g., system.debug_level -> MONKEY_SYSTEM_DEBUG_LEVEL
                env_var = f"MONKEY_{section.upper()}_{key.upper()}"

                # Convert value to string
                if isinstance(value, bool):
                    env_value = "1" if value else "0"
                else:
                    env_value = str(value)

                # Set environment variable
                os.environ[env_var] = env_value
                debug(f"Exported config to environment: {env_var}={env_value}")