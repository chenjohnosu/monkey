"""
Simplified logging functionality for the document analysis toolkit
"""

import datetime
import logging
import sys
import os
from typing import Optional, Any

class LogManager:
    """Centralized logging management"""

    # Store file handler for batch mode
    file_handler = None

    @classmethod
    def configure(cls, level=logging.INFO):
        """Configure the root logger with standard formatting"""
        root_logger = logging.getLogger()

        # Clear existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set level
        root_logger.setLevel(level)

        # Add console handler
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

        # Silence overly verbose libraries
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

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
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        return level

    @classmethod
    def add_file_handler(cls, log_file):
        """Add a file handler for batch mode logging"""
        if cls.file_handler:
            # Remove existing file handler
            root_logger = logging.getLogger()
            root_logger.removeHandler(cls.file_handler)

        # Create directory if needed
        log_dir = os.path.dirname(log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create new file handler
        cls.file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                     datefmt='%Y-%m-%d %H:%M:%S')
        cls.file_handler.setFormatter(formatter)

        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(cls.file_handler)

        return cls.file_handler

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
            if debug_level != 'debug':
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

# Initialize logging system
LogManager.configure()