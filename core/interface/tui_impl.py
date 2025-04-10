"""
Monkey TUI - Terminal User Interface for Monkey Document Analysis Toolkit
Main output on left, system messages on right, command input at bottom
"""

from textual.app import App
from textual.widgets import Header, Footer, Input, Static, Label, RichLog
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.widgets import Static
from textual import events
from textual.message import Message
from textual.binding import Binding
import threading
import shlex
import time
import queue
import logging
import importlib
import sys
import os
import builtins
from datetime import datetime

# Add root directory to path for importing from project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import improved logging
from core.engine.logging import get_logger, error, warning, info, debug, trace

# Get a logger for this module
logger = get_logger("tui_impl")

class StatusBar(Static):
    """Status bar for Monkey"""

    def __init__(self, config):
        # Disable markup rendering to avoid special character issues
        super().__init__("Status", markup=False)
        self.config = config
        self.command_processor = None
        self.query_mode = False
        self.workspace = config.get('workspace.default', 'none')
        self.llm_model = config.get('llm.default_model', 'none')
        self.embedding_model = config.get('embedding.default_model', 'none')  # Added embedding model
        self.update_status()

    def update_status(self):
        """Update the status information"""
        # Get current LLM model from config
        self.llm_model = self.config.get('llm.default_model', 'none')

        # Get current embedding model from config
        self.embedding_model = self.config.get('embedding.default_model', 'none')  # Get embedding model

        # Get current workspace from command processor if available
        if self.command_processor and hasattr(self.command_processor, 'current_workspace'):
            self.workspace = self.command_processor.current_workspace

        # Update query mode based on command processor's query engine
        if self.command_processor and hasattr(self.command_processor, 'query_engine'):
            try:
                self.query_mode = self.command_processor.query_engine.is_active()
            except Exception:
                # Fallback if query_engine doesn't have is_active method
                pass

        # Format the status text - now including embedding model
        mode = "query" if self.query_mode else "command"
        self.update(f"[ workspace: {self.workspace} | llm: {self.llm_model} | embed: {self.embedding_model} | mode: {mode} ]")


class OutputLog(RichLog):
    """Main output area using RichLog widget for better scrolling"""

    def clear(self):
        """Clear the log output"""
        super().clear()  # Call the parent class's clear method


class SystemLog(RichLog):
    """Dedicated logging area for system messages"""

    def __init__(self):
        super().__init__(highlight=True, markup=True, auto_scroll=True)
        # Add a title to the log
        super().write("[bold]System Messages[/bold]")

    # No custom write method - use the parent class implementation


# Modified input handler for interactive CLI commands
class TUIInputHandler:
    """Custom input handler to intercept input() calls from CLI functions"""

    def __init__(self, app):
        self.app = app
        self.input_requested = False
        self.input_prompt = ""
        self.input_queue = []
        self.callback = None

    def input(self, prompt=""):
        """Simulate the input() function but route through TUI"""
        self.input_requested = True
        self.input_prompt = prompt

        # Display the prompt in the output log
        self.app.query_one(OutputLog).write(prompt)

        # Create an event to wait for input
        event = threading.Event()

        # Set callback to be called when input is received
        def input_callback(value):
            self.input_queue.append(value)
            event.set()

        self.callback = input_callback

        # Show the input field for user response
        self.app.query_one("#command_input").placeholder = f"Respond to: {prompt}"

        # Wait for input (will be provided by the on_input_submitted method)
        event.wait()

        # Reset the callback and input_requested flag
        self.callback = None
        self.input_requested = False
        self.app.query_one("#command_input").placeholder = "Enter command..."

        # Return the value that was entered
        return self.input_queue.pop(0)


# Custom logger handler to redirect logs to the TUI
class TUILogHandler(logging.Handler):
    """Custom logging handler that redirects logs to the TUI system log area"""

    def __init__(self, app):
        super().__init__()
        self.app = app
        self.log_queue = queue.Queue()
        self.formatter = logging.Formatter('%(levelname)s: %(message)s')

    def emit(self, record):
        """Emit a log record to the TUI"""
        try:
            # Format the log message
            msg = self.formatter.format(record)

            # Add timestamp to beginning of message
            timestamp = datetime.now().strftime('%H:%M:%S')

            # Process the message with proper wrapping
            # Add an indent marker at the beginning of each line except the first
            lines = msg.split('\n')
            formatted_lines = []

            # Format the first line with timestamp
            if lines:
                first_line = f"[{timestamp}] {lines[0]}"

                # Add log level colors
                if record.levelno >= logging.ERROR:
                    formatted_lines.append(f"[bold red]{first_line}[/bold red]")
                elif record.levelno >= logging.WARNING:
                    formatted_lines.append(f"[yellow]{first_line}[/yellow]")
                elif record.levelno >= logging.INFO:
                    formatted_lines.append(f"[white]{first_line}[/white]")
                else:  # DEBUG or TRACE
                    formatted_lines.append(f"[dim]{first_line}[/dim]")

                # Format continuation lines with proper indentation and same color
                indent = "          "  # 10 spaces for indentation

                for i in range(1, len(lines)):
                    if record.levelno >= logging.ERROR:
                        formatted_lines.append(f"[bold red]{indent}{lines[i]}[/bold red]")
                    elif record.levelno >= logging.WARNING:
                        formatted_lines.append(f"[yellow]{indent}{lines[i]}[/yellow]")
                    elif record.levelno >= logging.INFO:
                        formatted_lines.append(f"[white]{indent}{lines[i]}[/white]")
                    else:  # DEBUG or TRACE
                        formatted_lines.append(f"[dim]{indent}{lines[i]}[/dim]")

            # Join the formatted lines and add to the queue
            formatted_msg = "\n".join(formatted_lines)
            self.log_queue.put(formatted_msg)

        except Exception:
            self.handleError(record)


class MonkeyTUI(App):
    """Main TUI application for Monkey"""

    CSS = """
        StatusBar {
            height: 1;
            width: 100%;
            background: #333344;
            color: #e0e0e0;
            padding: 0 1;
        }

        /* Main container with horizontal layout */
        #main_container {
            width: 100%;
            height: 100%;
            layout: horizontal;
        }

        /* Left panel container - vertical layout */
        #left_panel {
            width: 70%;
            height: 100%;
            layout: vertical;
        }

        /* Main output area - in the middle of the left panel */
        #scrollable_output {
            width: 100%;
            height: 1fr;  /* Takes available space */
        }

        OutputLog {
            height: 100%;
            background: #1e1e2e;
            color: #cdd6f4;
        }

        /* Right side container for system log */
        #system_log_container {
            width: 30%;
            height: 100%;
            border-left: solid #666666;
        }

        SystemLog {
            height: 100%;
            background: #282a36;
            color: #f8f8f2;
            padding: 0 1;
        }

        Input {
            height: 3;
            background: #313244;
            color: #cdd6f4;
            border: none;
            padding: 0 1;
        }

        Input:focus {
            border: none;  /* Remove focus border */
        }

        ScrollableContainer:focus {
            border: none;  /* Remove focus border */
        }
        """

    def __init__(self, command_processor):
        super().__init__()
        self.command_processor = command_processor
        self.config = command_processor.config
        self.input_handler = TUIInputHandler(self)
        self.command_running = False
        self.current_command = None
        self.system_log = None
        self.log_handler = TUILogHandler(self)
        self.query_mode = False  # Track query mode status

        # Set up log handler for debug_print
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)

        # Set up timer for log updates
        self._timer_running = False

        # Set up periodic status update timer
        self._status_timer_running = False

        # Store reference to debug_print for monitoring
        from core.engine.logging import debug_print
        self.original_debug_print = debug_print

        # Override debug_print to capture to system log
        def tui_debug_print(config, message):
            # Call the original function
            self.original_debug_print(config, message)
            # Also capture to our log
            self.log_handler.log_queue.put(f"[dim]DEBUG: {message}[/dim]")

        # Replace the debug_print function
        import core.engine.logging
        core.engine.logging.debug_print = tui_debug_print

    def compose(self):
        """Compose the TUI layout with vertical split first, then horizontal divisions on left side"""

        # Create a horizontal split for the main area (left/right panels)
        with Horizontal(id="main_container"):
            # Left panel container (status, output, command input)
            with Vertical(id="left_panel"):
                # Status bar at the top
                status_bar = StatusBar(self.config)
                # Connect the command processor to the status bar
                status_bar.command_processor = self.command_processor
                yield status_bar

                # Main output area in the middle (scrollable)
                with ScrollableContainer(id="scrollable_output"):
                    yield OutputLog(auto_scroll=True, highlight=True)

                # Command input at the bottom
                yield Input(placeholder="Enter command...", id="command_input")

            # Right panel for system log (keeps the same)
            with ScrollableContainer(id="system_log_container"):
                yield SystemLog()

    def on_mount(self):
        """Handle app mount event"""
        # Signal that TUI is ready to display messages
        from core.engine.logging import LogManager
        LogManager.set_tui_ready(True)

        logger.info("TUI interface mounted")
        self.query_one("#command_input").focus()

        # Store reference to system log
        self.system_log = self.query_one(SystemLog)

        # Connect command processor to status bar
        status_bar = self.query_one(StatusBar)
        status_bar.command_processor = self.command_processor

        # Force an initial status update
        status_bar.update_status()

        # Start log update timer
        self._start_log_timer()

        # Start status update timer
        self._start_status_timer()

        # Log that the TUI is ready
        logger.info("TUI interface ready")

    def _check_query_mode(self):
        """
        Check the query mode status from the command processor and update the UI
        """
        try:
            # Get direct reference to check if query engine is active
            new_query_mode = False

            if hasattr(self.command_processor, 'query_engine'):
                # First try the query_engine method
                if hasattr(self.command_processor.query_engine, 'is_active'):
                    new_query_mode = self.command_processor.query_engine.is_active()
                # Fallback 1: Check if query_engine has an "active" attribute
                elif hasattr(self.command_processor.query_engine, 'active'):
                    new_query_mode = bool(self.command_processor.query_engine.active)
                # Last resort: Check if query mode is active directly on command processor
                elif hasattr(self.command_processor, 'query_mode'):
                    new_query_mode = bool(self.command_processor.query_mode)

            # Only update if there's a change to avoid unnecessary UI refreshes
            if new_query_mode != self.query_mode:
                # Update our internal state
                self.query_mode = new_query_mode

                # Update status bar
                status_bar = self.query_one(StatusBar)
                status_bar.query_mode = self.query_mode
                self.call_from_thread(lambda: status_bar.update_status())

                # Log the mode change
                logger.info(f"Mode changed to: {'query' if self.query_mode else 'command'}")
        except Exception as e:
            # Don't let errors in mode detection crash the app
            logger.error(f"Error detecting query mode: {str(e)}")

    def _start_log_timer(self):
        """Start a timer to periodically update logs"""
        if self._timer_running:
            return

        self._timer_running = True

        # Create a daemon thread to process logs periodically
        def update_logs_thread():
            while self._timer_running:
                try:
                    # Check for logs to display
                    self.update_logs()
                    # Sleep a short time
                    time.sleep(0.1)
                except Exception as e:
                    # Log errors in the timer thread
                    print(f"Error in log timer thread: {str(e)}")

        # Start the thread as a daemon so it doesn't block app exit
        thread = threading.Thread(target=update_logs_thread, daemon=True)
        thread.start()

    def _start_status_timer(self):
        """Start a timer to periodically update status bar"""
        if self._status_timer_running:
            return

        self._status_timer_running = True

        # Create a daemon thread for periodic status updates
        def update_status_thread():
            while self._status_timer_running:
                try:
                    # Check if the workspace has changed in command processor
                    if (hasattr(self.command_processor, 'current_workspace') and
                        self.query_one(StatusBar).workspace != self.command_processor.current_workspace):
                        # Force an update to reflect the new workspace
                        self.call_from_thread(lambda: self.query_one(StatusBar).update_status())

                    # Check query mode status
                    self._check_query_mode()

                    # Update the status bar
                    self.call_from_thread(lambda: self.query_one(StatusBar).update_status())

                    # Sleep longer than log timer - status doesn't need to update as frequently
                    time.sleep(0.5)
                except Exception as e:
                    # Log errors in the timer thread
                    print(f"Error in status timer thread: {str(e)}")

        # Start the thread as a daemon so it doesn't block app exit
        thread = threading.Thread(target=update_status_thread, daemon=True)
        thread.start()

    def update_logs(self):
        """Update the system log area with any queued log messages"""
        if not self.system_log:
            return

        # Process queued logs
        try:
            # Get up to 10 messages at a time to prevent UI freezing
            for _ in range(10):
                try:
                    # Get message with short timeout
                    msg = self.log_handler.log_queue.get(block=False)

                    # Display in system log area
                    self.call_from_thread(lambda m=msg: self.system_log.write(m))

                    # Mark as done
                    self.log_handler.log_queue.task_done()
                except queue.Empty:
                    # No more messages
                    break
        except Exception as e:
            # Catch any errors in log processing
            print(f"Error updating logs: {str(e)}")
            import traceback
            print(traceback.format_exc())

    def on_input_submitted(self, event):
        """Handle command input"""
        value = event.value
        input_widget = self.query_one(Input)
        input_widget.value = ""

        # If there's an input request pending, handle it through the input handler
        if self.input_handler.input_requested and self.input_handler.callback:
            log = self.query_one(OutputLog)
            log.write(value)  # Echo user's response
            self.input_handler.callback(value)
            return

        # Normal command processing
        if not value:
            return

        # Prevent running multiple commands simultaneously
        if self.command_running:
            self.query_one(OutputLog).write("Command already running. Please wait...")
            return

        self.command_running = True
        self.current_command = value

        # Show the command being executed
        log = self.query_one(OutputLog)
        log.write(f"> {value}")

        # Run the command in a separate thread to prevent UI freezing
        def run_command():
            try:
                # Process the command and update everything
                self._process_command(value)

                # Immediately capture the current workspace
                if hasattr(self.command_processor, 'current_workspace'):
                    current_ws = self.command_processor.current_workspace
                    # Force status bar update with the new workspace
                    self.call_from_thread(lambda: self._force_status_update(current_ws))

                # Check query mode status after command executes
                self._check_query_mode()
            except Exception as e:
                logger.error(f"Error processing command: {str(e)}")
                import traceback
                logger.debug(traceback.format_exc())
                self.call_from_thread(lambda: log.write(f"Error processing command: {str(e)}"))
            finally:
                self.command_running = False
                self.current_command = None
                # Update status bar
                self.call_from_thread(lambda: self.query_one(StatusBar).update_status())

        threading.Thread(target=run_command).start()

    def _force_status_update(self, workspace):
        """Force status bar update with current workspace"""
        status_bar = self.query_one(StatusBar)
        status_bar.workspace = workspace  # Directly set the workspace
        status_bar.update_status()  # Update display

    def _process_command(self, command_string):
        """Process a command through the CommandProcessor"""
        log = self.query_one(OutputLog)

        try:
            # Check for direct quit/exit commands - handle these immediately
            if command_string.lower() in ['/quit', '/exit']:
                # Stop timers first
                self._timer_running = False
                self._status_timer_running = False
                # Exit directly using the same method as Ctrl-Q
                self.exit()
                return

            # Detect load command to update workspace immediately
            load_cmd = False
            if command_string.startswith('/load'):
                load_cmd = True
                tokens = shlex.split(command_string)
                if len(tokens) > 1:
                    # Prepare for workspace change
                    self.query_one(StatusBar).update_status()

            # Create a string capture class to redirect output
            class StringCapture:
                def __init__(self, app):
                    self.app = app
                    self.log_widget = app.query_one(OutputLog)
                    self.line_buffer = ""

                def write(self, text):
                    # Always preserve \r\n in the text
                    if '\r\n' in text:
                        text = text.replace('\r\n', '\n')

                    # Handle complete lines (with newlines)
                    if '\n' in text:
                        lines = text.split('\n')
                        # Process all lines except the last one (which might be incomplete)
                        for i in range(len(lines) - 1):
                            line = self.line_buffer + lines[i]
                            if line:  # Only add non-empty lines
                                self.app.call_from_thread(
                                    lambda l=line: self.log_widget.write(l)
                                )
                            self.line_buffer = ""
                        # The last part might be an incomplete line
                        self.line_buffer += lines[-1]
                    else:
                        # Accumulate partial line
                        self.line_buffer += text

                    # If we have a complete line in buffer, output it
                    if self.line_buffer.endswith('\r'):
                        line = self.line_buffer[:-1]  # Remove the \r
                        if line:
                            self.app.call_from_thread(
                                lambda l=line: self.log_widget.write(l)
                            )
                        self.line_buffer = ""

                def flush(self):
                    # Output any remaining text in the buffer
                    if self.line_buffer:
                        self.app.call_from_thread(
                            lambda l=self.line_buffer: self.log_widget.write(l)
                        )
                        self.line_buffer = ""

            # Override built-in input function for interactive commands
            original_input = builtins.input
            builtins.input = self.input_handler.input

            # Temporarily redirect stdout to capture CLI output
            import sys
            original_stdout = sys.stdout
            string_capture = StringCapture(self)
            sys.stdout = string_capture

            # Process the command with CommandProcessor
            # Check if it's a quit command to exit the app
            if command_string.startswith('/quit') or command_string.startswith('/exit'):
                # First run the actual command
                self.command_processor.process_command(command_string)
                # Then exit the app
                self._timer_running = False
                self._status_timer_running = False
                self.call_from_thread(self.exit)
                return

            # Process the command
            self.command_processor.process_command(command_string)

            # If this was a load command, immediately update status
            if load_cmd and hasattr(self.command_processor, 'current_workspace'):
                # Force status bar to use the new workspace immediately
                status_bar = self.query_one(StatusBar)
                status_bar.workspace = self.command_processor.current_workspace
                self.call_from_thread(lambda: status_bar.update_status())

            # Update query mode status by directly checking query_engine
            if hasattr(self.command_processor, 'query_engine'):
                try:
                    self.query_mode = self.command_processor.query_engine.is_active()

                    # Also update the StatusBar.query_mode to ensure consistency
                    status_bar = self.query_one(StatusBar)
                    status_bar.query_mode = self.query_mode

                    # Force status bar update
                    self.call_from_thread(lambda: status_bar.update_status())
                except Exception as e:
                    logger.error(f"Error checking query mode: {str(e)}")

            # Flush any remaining output
            string_capture.flush()

        finally:
            # Restore stdout and input function
            sys.stdout = original_stdout
            builtins.input = original_input

            # Always update UI elements after any command to ensure status bar is current
            self.call_from_thread(lambda: self.query_one(StatusBar).update_status())

    def on_unmount(self):
        """Clean up resources when app is closing"""
        logger.info("TUI interface closing")

        # Stop the timers
        self._timer_running = False
        self._status_timer_running = False

        # Remove our log handler
        root_logger = logging.getLogger()
        if self.log_handler in root_logger.handlers:
            root_logger.removeHandler(self.log_handler)

        # Restore original debug_print
        import core.engine.logging
        core.engine.logging.debug_print = self.original_debug_print