"""
Centralized logging functionality for the document analysis toolkit
"""

import datetime
import logging
import queue
import sys
import re
import os
from typing import Any, Optional

# Message queues for TUI mode
message_queues = {
    'user': queue.Queue(),
    'status': queue.Queue(),
    'system': queue.Queue(),
}

# Global log manager instance
log_manager = None

class LogManager:
    """
    Unified logging and output manager for both CLI and TUI modes

    Provides:
    - Single entry point for all output
    - Interface-aware message routing
    - Consistent formatting
    - Configurable log levels
    """

    # Log level constants (compatible with standard logging)
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    # Message categories
    USER = "user"      # User-facing main output
    STATUS = "status"  # Status bar updates
    SYSTEM = "system"  # System messages

    def __init__(self, config=None):
        """Initialize the LogManager"""
        self.config = config
        self.tui_mode = False
        self.tui_ready = False
        self.log_level = self.INFO
        self.log_file = None
        self.log_file_handle = None
        self.message_queues = message_queues
        self._setup_logging()

    def _setup_logging(self):
        """Configure Python's standard logging to use our handler"""
        # Define TRACE level for logging
        logging.TRACE = self.TRACE
        logging.addLevelName(logging.TRACE, "TRACE")

        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.TRACE)  # Capture everything, filter later

        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add our custom handler
        handler = logging.StreamHandler(stream=sys.stderr)
        handler.setFormatter(logging.Formatter('%(message)s'))

        # Override the emit method to use our routing
        original_emit = handler.emit
        def custom_emit(record):
            msg = handler.format(record)
            if record.levelno >= logging.ERROR:
                self.error(msg)
            elif record.levelno >= logging.WARNING:
                self.warning(msg)
            elif record.levelno >= logging.INFO:
                self.info(msg)
            elif record.levelno >= logging.DEBUG:
                self.debug(msg)
            else:
                self.trace(msg)

        handler.emit = custom_emit
        root_logger.addHandler(handler)

        # Silence overly verbose libraries
        logging.getLogger('numba').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)

    def set_interface_mode(self, mode="cli"):
        """Set interface mode to 'cli' or 'tui'"""
        self.tui_mode = (mode.lower() == "tui")

    def set_log_level(self, level):
        """Set the minimum log level to display"""
        self.log_level = level

    def start_file_logging(self, filepath=None):
        """Enable logging to file"""
        if filepath:
            # Close existing file if open
            if self.log_file_handle:
                self.log_file_handle.close()

            try:
                # Ensure directory exists
                log_dir = os.path.dirname(filepath)
                if log_dir and not os.path.exists(log_dir):
                    os.makedirs(log_dir)

                # Open log file
                self.log_file_handle = open(filepath, 'w', encoding='utf-8')
                self.log_file = filepath

                # Log successful start
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.log_file_handle.write(f"=== Log started at {timestamp} ===\n")
                self.log_file_handle.flush()

                return True
            except Exception as e:
                # Fall back to direct sys.stderr write since logging might not be reliable
                sys.stderr.write(f"Error opening log file: {str(e)}\n")
                return False
        else:
            return False

    def stop_file_logging(self):
        """Disable logging to file"""
        if self.log_file_handle:
            try:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                self.log_file_handle.write(f"=== Log ended at {timestamp} ===\n")
                self.log_file_handle.close()
                result = self.log_file
                self.log_file = None
                self.log_file_handle = None
                return result
            except Exception:
                return None
        return None

    def user(self, message, format_type=None, **kwargs):
        """Output user-facing content"""
        self._log(self.USER, self.INFO, message, format_type, **kwargs)

    def status(self, message):
        """Update status information"""
        self._log(self.STATUS, self.INFO, message)

    def debug(self, message):
        """Log debug information"""
        self._log(self.SYSTEM, self.DEBUG, message)

    def info(self, message):
        """Log general information"""
        self._log(self.SYSTEM, self.INFO, message)

    def warning(self, message):
        """Log warning"""
        self._log(self.SYSTEM, self.WARNING, message)

    def error(self, message):
        """Log error"""
        self._log(self.SYSTEM, self.ERROR, message)

    def trace(self, message):
        """Log detailed trace information"""
        self._log(self.SYSTEM, self.TRACE, message)

    def _log(self, category, level, message, format_type=None, **kwargs):
        """Internal logging implementation"""
        # Skip logging if below current log level
        if level < self.log_level:
            return

        # Format the message based on category and level
        formatted = self._format_message(category, level, message, format_type, **kwargs)
        plain = self._strip_formatting(formatted)

        # Route based on interface mode and category
        if self.tui_mode and self.tui_ready:
            # Add to appropriate message queue for TUI
            self.message_queues[category].put(formatted)
        else:
            # Direct output for CLI mode
            self._output_to_cli(category, level, plain)

        # Always write to log file if enabled
        if self.log_file_handle:
            self._write_to_log_file(level, plain)

    def _format_message(self, category, level, message, format_type=None, **kwargs):
        """Format message with appropriate styling"""
        # Import formatting here to avoid circular imports
        from core.engine.formatting import Formatter

        # Get timestamp for context
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')

        # Format based on message category and level
        if category == self.USER:
            if format_type == 'header':
                return Formatter.format_header(message, for_tui=self.tui_mode)
            elif format_type == 'subheader':
                return Formatter.format_subheader(message, for_tui=self.tui_mode)
            elif format_type == 'mini_header':
                return Formatter.format_mini_header(message, for_tui=self.tui_mode)
            elif format_type == 'kv':
                key = kwargs.get('key', '')
                indent = kwargs.get('indent', 0)
                return Formatter.format_key_value(key, message, indent, for_tui=self.tui_mode)
            elif format_type == 'list':
                indent = kwargs.get('indent', 0)
                return Formatter.format_list_item(message, indent, for_tui=self.tui_mode)
            elif format_type == 'feedback':
                success = kwargs.get('success', True)
                return Formatter.format_feedback(message, success, for_tui=self.tui_mode)
            elif format_type == 'code':
                indent = kwargs.get('indent', 0)
                return Formatter.format_code_block(message, indent, for_tui=self.tui_mode)
            elif format_type == 'command':
                return Formatter.format_command(message, for_tui=self.tui_mode)
            else:
                # Default formatting for user messages
                return message

        elif category == self.SYSTEM:
            # Format system messages with level prefix
            level_name = logging.getLevelName(level)
            if level >= self.ERROR:
                return Formatter.format_error(f"[{timestamp}] {level_name}: {message}", for_tui=self.tui_mode)
            elif level >= self.WARNING:
                return Formatter.format_warning(f"[{timestamp}] {level_name}: {message}", for_tui=self.tui_mode)
            elif level >= self.INFO:
                return Formatter.format_info(f"[{timestamp}] {level_name}: {message}", for_tui=self.tui_mode)
            elif level >= self.DEBUG:
                return Formatter.format_debug(f"[{timestamp}] {level_name}: {message}", for_tui=self.tui_mode)
            else:
                return Formatter.format_trace(f"[{timestamp}] {level_name}: {message}", for_tui=self.tui_mode)

        else:  # STATUS or other categories
            return message

    def _strip_formatting(self, text):
        """Remove ANSI escape codes from text"""
        from core.engine.formatting import Formatter
        return Formatter.strip_ansi(text)

    def _output_to_cli(self, category, level, formatted_message):
        """Output to CLI"""
        if category == self.USER:
            # Main output goes to stdout
            print(formatted_message)
        else:
            # System messages and status updates go to stderr
            print(formatted_message, file=sys.stderr)

    def _write_to_log_file(self, level, message):
        """Write message to log file if enabled"""
        if self.log_file_handle:
            try:
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                level_name = logging.getLevelName(level)
                self.log_file_handle.write(f"[{timestamp}] {level_name}: {message}\n")
                self.log_file_handle.flush()
            except Exception:
                # If writing to log file fails, we can't really log the error
                # Just continue silently to avoid infinite recursion
                pass

    def get_queued_messages(self, category):
        """Get messages from queue for TUI processing"""
        messages = []
        queue = self.message_queues.get(category, None)

        if not queue:
            return messages

        # Get all messages without blocking
        while not queue.empty():
            try:
                messages.append(queue.get_nowait())
                queue.task_done()
            except queue.Empty:
                break

        return messages

# Deprecation wrapper for maintaining compatibility
def debug_print(config, message):
    """
    Legacy debug_print function - redirects to log_manager.debug

    Args:
        config: Configuration object (ignored, kept for compatibility)
        message: Message to log
    """
    global log_manager
    if log_manager:
        log_manager.debug(message)
    else:
        # Fall back to stderr if log_manager not initialized
        print(f"DEBUG: {message}", file=sys.stderr)

# Initialize global log manager if not already done
def get_log_manager(config=None):
    """Get or create the global log manager"""
    global log_manager
    if not log_manager:
        log_manager = LogManager(config)
    return log_manager

# Compatibility functions mapped to log_manager methods
def error(message):
    """Log an error message"""
    get_log_manager().error(message)

def warning(message):
    """Log a warning message"""
    get_log_manager().warning(message)

def info(message):
    """Log an info message"""
    get_log_manager().info(message)

def debug(message):
    """Log a debug message"""
    get_log_manager().debug(message)

def trace(message):
    """Log a trace message"""
    get_log_manager().trace(message)

def set_tui_mode(enabled=False):
    """Set whether we're in TUI mode"""
    get_log_manager().set_interface_mode("tui" if enabled else "cli")
    return enabled

def set_tui_ready(ready=True):
    """Set whether the TUI is fully initialized and ready for messages"""
    get_log_manager().tui_ready = ready
    return ready

def get_logger(name):
    """Get a logger instance - compatibility function"""
    return logging.getLogger(name)

# Initialize the log manager
log_manager = get_log_manager()

# Configure root logger
logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[])