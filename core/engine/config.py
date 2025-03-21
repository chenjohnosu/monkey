"""
Configuration management for the document analysis toolkit
"""

import os
import yaml
from core.engine.logging import debug_print

class Config:
    """Configuration management class"""
    
    def __init__(self, config_path='config.yaml'):
        """Initialize configuration from file"""
        self.config_path = config_path
        self.loaded_guides = {}
        self.version = '0.1.0'
        
        # Load configuration
        try:
            with open(config_path, 'r', encoding='utf-8') as file:
                self.config = yaml.safe_load(file)
                if not self.config:
                    self.config = {}
        except FileNotFoundError:
            print(f"Config file not found: {config_path}. Using default configuration.")
            self.config = {}
        except yaml.YAMLError as e:
            print(f"Error parsing config file: {str(e)}. Using default configuration.")
            self.config = {}
        
        # Initialize with defaults if needed
        self._ensure_defaults()
        
        # Debug output
        if self.get('system.debug'):
            debug_print(self, f"Configuration loaded from {config_path}")
    
    def _ensure_defaults(self):
        """Ensure default configuration values are set"""
        defaults = {
            'system': {
                'debug': False,
                'output_format': 'txt'
            },
            'hardware': {
                'use_cuda': 'auto',
                'use_mps': 'auto',
                'device': 'auto'
            },
            'llm': {
                'default_model': 'mistral',
                'ollama_host': 'http://localhost',
                'ollama_port': 11434
            },
            'embedding': {
                'default_model': 'mixbread'
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
                'vector_store': 'chroma'
            }
        }
        
        # Ensure all default sections exist
        for section, values in defaults.items():
            if section not in self.config:
                self.config[section] = {}
            
            # Ensure all default values exist in each section
            for key, value in values.items():
                if key not in self.config[section]:
                    self.config[section][key] = value
    
    def get(self, path, default=None):
        """Get a configuration value by path (e.g., 'system.debug')"""
        sections = path.split('.')
        config = self.config
        
        for section in sections:
            if section not in config:
                return default
            config = config[section]
        
        return config
    
    def set(self, path, value):
        """Set a configuration value by path (e.g., 'system.debug')"""
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
        self._save_config()
        
        # Debug output
        if self.get('system.debug'):
            debug_print(self, f"Configuration updated: {path} = {value}")
    
    def _save_config(self):
        """Save the configuration to file"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as file:
                yaml.dump(self.config, file, default_flow_style=False)
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
    
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
