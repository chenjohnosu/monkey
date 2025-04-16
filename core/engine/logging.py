"""
Centralized logging functionality for the document analysis toolkit
"""

import datetime
import logging
import queue
import sys
import re
from typing import Any, Optional

# Message queue for TUI mode
tui_message_queue = queue.Queue()
# Flag to indicate if we're in TUI mode
_in_tui_mode = False
# Flag to indicate if TUI is fully initialized
_tui_ready = False
# Buffer for messages that arrive before TUI is ready
_pre_tui_buffer = queue.Queue()



logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")


class FormattedLogHandler(logging.Handler):
    """Custom handler that formats logs consistently for console and TUI"""

    def __init__(self):
        super().__init__()
        self.tui_mode = False  # Track TUI mode status

    def set_tui_mode(self, enabled=False):
        """Set whether we're in TUI mode"""
        global _in_tui_mode
        self.tui_mode = enabled
        _in_tui_mode = enabled

        # If disabling TUI mode, ensure any buffered messages are processed
        if not enabled:
            while not _pre_tui_buffer.empty():
                msg = _pre_tui_buffer.get()
                print(msg, file=sys.stderr)

    def emit(self, record):
        try:
            # Format the basic message
            msg = self.format(record)

            # Check if we're in TUI mode and TUI is not ready
            if self.tui_mode and not _tui_ready:
                # Buffer messages before TUI is ready
                _pre_tui_buffer.put(msg)
                return

            # Determine message color and formatting for TUI or console
            if self.tui_mode:
                color_map = {
                    logging.DEBUG: "[dim]DEBUG[/dim]",
                    logging.INFO: "[white]INFO[/white]",
                    logging.WARNING: "[yellow]WARNING[/yellow]",
                    logging.ERROR: "[bold red]ERROR[/bold red]",
                    logging.CRITICAL: "[bold red]CRITICAL[/bold red]"
                }

                # TUI-specific formatting with colored levels and timestamp
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                level_display = color_map.get(record.levelno, f"[{record.levelname}]")

                # Format the message with timestamp and level
                lines = msg.split('\n')
                if len(lines) > 1:
                    formatted_lines = [f"[{timestamp}] {level_display}: {lines[0]}"]
                    formatted_lines.extend([f"          {line}" for line in lines[1:]])
                    formatted = '\n'.join(formatted_lines)
                else:
                    formatted = f"[{timestamp}] {level_display}: {msg}"

                # Add to TUI message queue if TUI is ready
                if _tui_ready:
                    tui_message_queue.put(formatted)
                # Otherwise, continue to stderr
                print(formatted, file=sys.stderr)
            else:
                # Simple console output without markup or timestamp
                print(msg, file=sys.stderr)
        except Exception:
            self.handleError(record)

def set_tui_ready(ready=True):
    """
    Set whether the TUI is fully initialized and ready for messages

    Args:
        ready: Whether TUI is ready
    """
    global _tui_ready
    _tui_ready = ready

    # If TUI is now ready, transfer any buffered messages to the main queue
    if ready:
        while not _pre_tui_buffer.empty():
            tui_message_queue.put(_pre_tui_buffer.get())

    return _tui_ready

def configure_root_logger(level=logging.INFO):
    """Configure the root logger with standard formatting"""
    root_logger = logging.getLogger()

    # Clear existing handlers to avoid duplicates
    if root_logger.handlers:
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

    # Set level
    root_logger.setLevel(level)

    # Add our custom handler
    custom_handler = FormattedLogHandler()
    formatter = logging.Formatter('%(message)s')
    custom_handler.setFormatter(formatter)
    root_logger.addHandler(custom_handler)

    # Silence overly verbose libraries
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)

    return root_logger

def get_tui_message():
    """
    Get the next message from the TUI message queue

    Returns:
        str: The next message, or None if queue is empty
    """
    if tui_message_queue.empty():
        return None
    return tui_message_queue.get()

def get_logger(name):
    """
    Get a logger instance with the specified name

    Args:
        name: Logger name (usually module name or component)

    Returns:
        logging.Logger: Configured logger
    """
    return logging.getLogger(name)

def debug_print(config, message):
    """
    Print a debug message if debug mode is enabled
    Redirects to standard logging

    Args:
        config: Configuration object with system.debug setting
        message: Message to print
    """
    # Check if debug is enabled in config
    debug_mode = False
    if config:
        try:
            debug_mode = config.get('system.debug', False)
        except Exception:
            pass

    if debug_mode:
        logging.debug(message)

def error(message):
    """
    Log an error message

    Args:
        message: Message to log
    """
    logging.error(message)

def warning(message):
    """
    Log a warning message

    Args:
        message: Message to log
    """
    logging.warning(message)

def info(message):
    """
    Log an info message

    Args:
        message: Message to log
    """
    logging.info(message)

def debug(message):
    """
    Log a debug message if debug is enabled

    Args:
        message: Message to log
    """
    logging.debug(message)

def trace(message):
    """
    Log a trace message (more detailed than debug)

    Args:
        message: Message to log
    """
    logging.log(logging.TRACE, message)

# Now define LogManager, which can reference the already defined functions
class LogManager:
    """
    Compatibility wrapper for logging functions to maintain existing code structure
    """
    _debug_enabled = False
    _log_handler = None

    @classmethod
    def _get_log_handler(cls):
        """Get the first FormattedLogHandler from root logger"""
        if not cls._log_handler:
            root_logger = logging.getLogger()
            for handler in root_logger.handlers:
                if isinstance(handler, FormattedLogHandler):
                    cls._log_handler = handler
                    break
        return cls._log_handler

    @classmethod
    def set_tui_mode(cls, enabled=False):
        """Set TUI mode for logging"""
        handler = cls._get_log_handler()
        if handler:
            handler.set_tui_mode(enabled)
        return enabled

    # Keep other existing methods the same
    @staticmethod
    def set_tui_ready(ready=True):
        """Signal if TUI is ready - kept for compatibility"""
        return ready

    @staticmethod
    def get_tui_message():
        """Placeholder for TUI message retrieval"""
        return None

    @classmethod
    def set_debug(cls, enabled=False):
        """Set debug mode"""
        cls._debug_enabled = enabled
        root_logger = logging.getLogger()

        if enabled:
            root_logger.setLevel(logging.DEBUG)
        else:
            if root_logger.level == logging.DEBUG:
                root_logger.setLevel(logging.INFO)

        return enabled

    # Attach module-level functions as static methods
    get_logger = staticmethod(get_logger)
    debug_print = staticmethod(debug_print)
    error = staticmethod(error)
    warning = staticmethod(warning)
    info = staticmethod(info)
    debug = staticmethod(debug)
    trace = staticmethod(trace)

# Initialize the logging system
configure_root_logger()