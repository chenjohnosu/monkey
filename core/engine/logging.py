"""
Simplified logging functionality for the document analysis toolkit
"""

import datetime
import logging
import sys
import os
import json
from typing import Optional, Any


class LogManager:
    """Centralized logging management"""

    # Store file handler for batch mode
    file_handler = None
    initialized = False
    # Default log file path in logs directory
    default_log_file = os.path.join('logs', 'monkey.log')
    log_file_path = default_log_file
    # Add a flag to track if logging is active
    logging_active = False  # Default: logging to console is OFF
    # Path to store logging state
    state_path = os.path.join('data', 'logging_state.json')

    @classmethod
    def configure(cls, level=logging.INFO, load_state=True, batch_mode=False):
        """Configure the root logger with standard formatting"""
        if cls.initialized:
            return logging.getLogger()

        # Create logs directory if it doesn't exist
        logs_dir = os.path.dirname(cls.default_log_file)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)

        # Load state if requested (first run)
        if load_state:
            cls._load_state()

        # If in batch mode, enable logging to both console and file
        if batch_mode:
            cls.logging_active = True
            cls.log_file_path = cls.default_log_file

        root_logger = logging.getLogger()

        # Clear existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set level
        root_logger.setLevel(level)

        # Add console handler if logging to console is active
        if cls.logging_active:
            handler = logging.StreamHandler(sys.stderr)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                         datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            root_logger.addHandler(handler)

        # Always add file handler to log to the default log file
        # Create file handler for the default log file
        cls.file_handler = logging.FileHandler(cls.log_file_path, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        cls.file_handler.setFormatter(formatter)

        # Always set file handler to DEBUG level to capture everything
        cls.file_handler.setLevel(logging.DEBUG)

        # Add to root logger
        root_logger.addHandler(cls.file_handler)

        # Silence overly verbose libraries
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

        cls.initialized = True

        # Save updated state
        cls._save_state()

        return root_logger

    @classmethod
    def set_level(cls, level_name):
        """Set logging level by name"""
        level_map = {
            'off': logging.CRITICAL + 100,  # A level higher than any defined level
            'error': logging.ERROR,
            'warning': logging.WARNING,
            'info': logging.INFO,
            'debug': logging.DEBUG,
            'trace': 5  # Custom TRACE level
        }

        level = level_map.get(level_name.lower(), logging.INFO)

        # Ensure logger is configured
        if not cls.initialized:
            cls.configure(level)
        else:
            root_logger = logging.getLogger()
            root_logger.setLevel(level)

        return level

    @classmethod
    def add_file_handler(cls, log_file):
        """Add a file handler for batch mode logging"""
        # Ensure logger is configured
        if not cls.initialized:
            cls.configure()

        root_logger = logging.getLogger()

        if cls.file_handler:
            # Remove existing file handler
            root_logger.removeHandler(cls.file_handler)
            cls.file_handler.close()
            cls.file_handler = None

        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create new file handler - ensure it's always set to DEBUG level
        # to capture all messages regardless of console log level
        cls.file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        cls.file_handler.setFormatter(formatter)

        # Always set file handler to DEBUG level to capture everything
        cls.file_handler.setLevel(logging.DEBUG)

        # Add to root logger
        root_logger.addHandler(cls.file_handler)

        # Store the log file path
        cls.log_file_path = log_file

        # Save state
        cls._save_state()

        return cls.file_handler

    @classmethod
    def redirect_all_logs(cls, log_file):
        """
        Redirect all logging to a file by removing console handlers
        and adding only a file handler

        Args:
            log_file (str): Path to the log file

        Returns:
            FileHandler: The created file handler
        """
        # Ensure logger is configured
        if not cls.initialized:
            cls.configure()

        root_logger = logging.getLogger()

        # Remove all existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            if hasattr(handler, 'close'):
                handler.close()

        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create file handler
        cls.file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        cls.file_handler.setFormatter(formatter)

        # Set to DEBUG level to capture everything
        cls.file_handler.setLevel(logging.DEBUG)

        # Add to root logger
        root_logger.addHandler(cls.file_handler)

        # Store the log file path
        cls.log_file_path = log_file

        # Save state
        cls._save_state()

        return cls.file_handler

    @classmethod
    def toggle_logging(cls, active=None):
        """
        Toggle logging on or off

        Args:
            active (bool, optional): If provided, set logging to this state,
                                    otherwise toggle the current state

        Returns:
            bool: The new state of logging (True if active, False if inactive)
        """
        # If a specific state is requested, set it
        if active is not None:
            cls.logging_active = bool(active)
        else:
            # Otherwise toggle the current state
            cls.logging_active = not cls.logging_active

        # Reconfigure the logger
        if cls.initialized:
            root_logger = logging.getLogger()

            # Remove all existing handlers
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)
                if hasattr(handler, 'close'):
                    handler.close()

            # Add console handler if logging is active
            if cls.logging_active:
                handler = logging.StreamHandler(sys.stderr)
                formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                            datefmt='%Y-%m-%d %H:%M:%S')
                handler.setFormatter(formatter)
                root_logger.addHandler(handler)

            # Always add a file handler for the log file
            cls.file_handler = logging.FileHandler(cls.log_file_path, mode='a', encoding='utf-8')
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                        datefmt='%Y-%m-%d %H:%M:%S')
            cls.file_handler.setFormatter(formatter)
            cls.file_handler.setLevel(logging.DEBUG)
            root_logger.addHandler(cls.file_handler)

        # Save state
        cls._save_state()

        return cls.logging_active

    @classmethod
    def get_log_file_path(cls):
        """Return the current log file path"""
        return cls.log_file_path

    @classmethod
    def get_logging_state(cls):
        """Return the current logging state"""
        return {
            'active': cls.logging_active,
            'log_file': cls.log_file_path
        }

    @classmethod
    def _save_state(cls):
        """Save the current logging state to a file"""
        # Create directory if needed
        state_dir = os.path.dirname(cls.state_path)
        if state_dir and not os.path.exists(state_dir):
            os.makedirs(state_dir)

        state = {
            'active': cls.logging_active,
            'log_file': cls.log_file_path
        }

        try:
            with open(cls.state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            # Can't log here to avoid infinite recursion
            print(f"Error saving logging state: {str(e)}")

    @classmethod
    def _load_state(cls):
        """Load the logging state from a file"""
        try:
            if os.path.exists(cls.state_path):
                with open(cls.state_path, 'r', encoding='utf-8') as f:
                    state = json.load(f)

                # Extract state variables
                cls.logging_active = state.get('active', False)  # Default to OFF
                cls.log_file_path = state.get('log_file', cls.default_log_file)
        except Exception as e:
            # Can't log here to avoid infinite recursion
            print(f"Error loading logging state: {str(e)}")
            # Use defaults
            cls.logging_active = False  # Default to OFF
            cls.log_file_path = cls.default_log_file

    @staticmethod
    def get_logger(name):
        """Get a logger with the specified name"""
        return logging.getLogger(name)


# Add TRACE level to the logging module
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")

def debug(message, config=None):
    """
    Log a debug message, respecting config settings if provided

    Args:
        message (str): The message to log
        config (Any, optional): Configuration object that may contain debug settings
    """
    # Check if we should respect config settings
    if config is not None:
        try:
            debug_level = config.get('system.debug_level', 'info')
            if debug_level not in ('debug', 'trace'):
                return  # Skip logging if debug not enabled
        except (AttributeError, KeyError):
            pass

    # Log the message
    logging.debug(message)

# Convenience functions
def error(message): logging.error(message)
def warning(message): logging.warning(message)
def info(message): logging.info(message)
def trace(message): logging.log(logging.TRACE, message)

# Initialize logging system with logging OFF by default and load state
# but don't initialize with batch mode yet - that will be set in monkey.py
LogManager.configure(load_state=True)