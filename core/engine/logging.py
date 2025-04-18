"""
Centralized logging functionality for the document analysis toolkit
"""

import datetime
import logging
import queue
import sys
from typing import Optional, Any

# Message queue for TUI mode
tui_message_queue = queue.Queue()
# Buffer for messages that arrive before TUI is ready
_pre_tui_buffer = queue.Queue()

# Global state flags
_in_tui_mode = False
_tui_ready = False

# Add TRACE level to the logging module
logging.TRACE = 5
logging.addLevelName(logging.TRACE, "TRACE")

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

        # Add our custom handler
        handler = FormattedLogHandler()
        formatter = logging.Formatter('%(message)s')
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

    @classmethod
    def set_tui_mode(cls, enabled=False):
        """Set whether we're in TUI mode"""
        global _in_tui_mode
        _in_tui_mode = enabled

        # Update handlers
        for handler in logging.getLogger().handlers:
            if isinstance(handler, FormattedLogHandler):
                handler.tui_mode = enabled

        return enabled

    @classmethod
    def set_tui_ready(cls, ready=True):
        """Set whether the TUI is fully initialized and ready for messages"""
        global _tui_ready
        _tui_ready = ready

        # If TUI is now ready, transfer buffered messages to main queue
        if ready:
            while not _pre_tui_buffer.empty():
                tui_message_queue.put(_pre_tui_buffer.get())

        return ready

    @staticmethod
    def get_logger(name):
        """Get a logger with the specified name"""
        return logging.getLogger(name)

    @staticmethod
    def get_tui_message():
        """Get the next message from the TUI message queue"""
        if tui_message_queue.empty():
            return None
        return tui_message_queue.get()

class FormattedLogHandler(logging.Handler):
    """Custom handler that formats logs consistently for console and TUI"""

    def __init__(self):
        super().__init__()
        self.tui_mode = False

    def emit(self, record):
        try:
            message = self.format(record)

            # Route message based on mode
            if self.tui_mode:
                self._emit_tui(record, message)
            else:
                # Simple console output
                print(message, file=sys.stderr)
        except Exception:
            self.handleError(record)

    def _emit_tui(self, record, message):
        """Handle TUI mode message emission"""
        global _tui_ready, _pre_tui_buffer, tui_message_queue

        # Format for TUI with color
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        color_map = {
            logging.DEBUG: "[dim]DEBUG[/dim]",
            logging.INFO: "[white]INFO[/white]",
            logging.WARNING: "[yellow]WARNING[/yellow]",
            logging.ERROR: "[bold red]ERROR[/bold red]",
            logging.CRITICAL: "[bold red]CRITICAL[/bold red]"
        }
        level_display = color_map.get(record.levelno, f"[{record.levelname}]")

        # Format multi-line messages with proper indentation
        lines = message.split('\n')
        if len(lines) > 1:
            formatted = f"[{timestamp}] {level_display}: {lines[0]}\n"
            formatted += '\n'.join([f"          {line}" for line in lines[1:]])
        else:
            formatted = f"[{timestamp}] {level_display}: {message}"

        # Route to appropriate queue
        if _tui_ready:
            tui_message_queue.put(formatted)
        else:
            _pre_tui_buffer.put(formatted)

        # Also print to stderr for safety
        print(formatted, file=sys.stderr)

# Helper functions for common logging operations

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

# Legacy alias for backward compatibility
debug_print = debug