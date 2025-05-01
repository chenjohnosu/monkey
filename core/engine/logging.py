"""
Simplified logging functionality for the document analysis toolkit
"""

import datetime
import logging
import sys
from typing import Optional, Any

class LogManager:
    """Centralized logging management"""

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
    def set_debug(cls, enabled=False):
        """Set debug mode"""
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG if enabled else logging.INFO)
        return enabled

    @staticmethod
    def get_logger(name):
        """Get a logger with the specified name"""
        return logging.getLogger(name)


class FormattedLogHandler(logging.Handler):
    """Custom handler that formats logs consistently for console"""

    def __init__(self):
        super().__init__()

    def emit(self, record):
        try:
            message = self.format(record)
            print(message, file=sys.stderr)
        except Exception:
            self.handleError(record)


# Add TRACE level to the logging module
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")


def debug(message: str, config: Any = None):
    """
    Log a debug message, respecting config settings if provided

    Args:
        message (str): The message to log
        config (Any, optional): Configuration object that may contain debug settings
    """
    # Check if we should respect config settings
    if config is not None:
        try:
            debug_mode = config.get('system.debug', False)
            if not debug_mode:
                return  # Skip logging if debug mode is disabled in config
        except (AttributeError, KeyError):
            # If we can't access config properly, fall back to regular debug
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