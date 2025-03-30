"""
Centralized logging functionality for the document analysis toolkit
To be placed in core/engine/logging.py
"""

import datetime
import logging
from typing import Any, Optional


# Configure the root logger
def configure_root_logger(level=logging.INFO):
    """Configure the root logger with basic settings"""
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Set level
    root_logger.setLevel(level)

    # Create console handler with formatter
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    return root_logger


class LogManager:
    """Centralized logging manager for consistent logging across modules"""

    @staticmethod
    def get_logger(name):
        """
        Get a logger with the specified name

        Args:
            name: Logger name (usually module name)

        Returns:
            logging.Logger: Configured logger
        """
        logger = logging.getLogger(name)
        return logger

    @staticmethod
    def debug_print(config: Any, message: str) -> None:
        """
        Print a debug message if debug mode is enabled

        Args:
            config: Configuration object with system.debug setting
            message: Message to print
        """
        if not config or config.get('system.debug'):
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"[DEBUG {timestamp}] {message}")

    @staticmethod
    def format_debug(message: str) -> str:
        """
        Format debug message with timestamp

        Args:
            message: Message to format

        Returns:
            str: Formatted debug message
        """
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"[DEBUG {timestamp}] {message}"

    @staticmethod
    def error(message: str) -> None:
        """
        Log an error message

        Args:
            message: Message to log
        """
        logging.error(message)

    @staticmethod
    def warning(message: str) -> None:
        """
        Log a warning message

        Args:
            message: Message to log
        """
        logging.warning(message)

    @staticmethod
    def info(message: str) -> None:
        """
        Log an info message

        Args:
            message: Message to log
        """
        logging.info(message)

    @staticmethod
    def debug(message: str) -> None:
        """
        Log a debug message

        Args:
            message: Message to log
        """
        logging.debug(message)

    @staticmethod
    def trace(message: str) -> None:
        """
        Log a trace message (more detailed than debug)

        Args:
            message: Message to log
        """
        # Python's logging doesn't have TRACE level by default
        # We use a level number lower than DEBUG
        logging.log(5, message)  # 5 is lower than DEBUG (10)


# Add TRACE level to the logging module
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")


# For backwards compatibility, provide standalone functions that use LogManager
def debug_print(config: Any, message: str) -> None:
    """
    Print a debug message if debug mode is enabled

    Args:
        config: Configuration object with system.debug setting
        message: Message to print
    """
    LogManager.debug_print(config, message)


def format_debug(message: str) -> str:
    """
    Format debug message with timestamp

    Args:
        message: Message to format

    Returns:
        str: Formatted debug message
    """
    return LogManager.format_debug(message)


def error(message: str) -> None:
    """
    Log an error message

    Args:
        message: Message to log
    """
    LogManager.error(message)


def warning(message: str) -> None:
    """
    Log a warning message

    Args:
        message: Message to log
    """
    LogManager.warning(message)


def info(message: str) -> None:
    """
    Log an info message

    Args:
        message: Message to log
    """
    LogManager.info(message)


def debug(message: str) -> None:
    """
    Log a debug message

    Args:
        message: Message to log
    """
    LogManager.debug(message)


def trace(message: str) -> None:
    """
    Log a trace message (more detailed than debug)

    Args:
        message: Message to log
    """
    LogManager.trace(message)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (usually module name or component)

    Returns:
        logging.Logger: Configured logger
    """
    return LogManager.get_logger(name)