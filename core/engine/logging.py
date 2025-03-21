"""
Centralized logging functionality for the document analysis toolkit
To be placed in core/engine/logging.py
"""

import datetime
from typing import Any, Optional


class LogManager:
    """Centralized logging manager for consistent logging across modules"""

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
    def info(message: str) -> None:
        """
        Print an info message

        Args:
            message: Message to print
        """
        print(f"[INFO] {message}")

    @staticmethod
    def warning(message: str) -> None:
        """
        Print a warning message

        Args:
            message: Message to print
        """
        print(f"[WARNING] {message}")

    @staticmethod
    def error(message: str) -> None:
        """
        Print an error message

        Args:
            message: Message to print
        """
        print(f"[ERROR] {message}")


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