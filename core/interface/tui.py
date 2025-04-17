"""
Terminal User Interface for Monkey Document Analysis Toolkit
"""

#import logging
from core.engine.logging import error, warning, info, debug, trace
from core.engine.logging import LogManager

def run_tui(command_processor):
    """
    Run the Terminal User Interface with the given command processor

    Args:
        command_processor: The initialized CommandProcessor instance
    """
    #logger = LogManager.get_logger(__name__)

    try:
        # Set TUI mode for logging before importing
        LogManager.set_tui_mode(True)

        # Import the actual TUI implementation
        from core.interface.tui_impl import MonkeyTUI

        # Create and run the TUI app
        app = MonkeyTUI(command_processor)

        # Signal that TUI is ready, but only here, BEFORE we start the event loop
        # This ensures it happens exactly once, at the right time
        LogManager.set_tui_ready(True)

        # Start the event loop
        app.run()
    except ImportError as e:
        # Reset logging mode
        LogManager.set_tui_mode(False)

        error(f"Textual library not installed. {e}")
        error("Please install it with: pip install textual")
        error("Falling back to CLI mode.")
        command_processor.start()
    except Exception as e:
        # Reset logging mode
        LogManager.set_tui_mode(False)

        error(f"Error initializing TUI: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        error("Falling back to CLI mode.")
        command_processor.start()