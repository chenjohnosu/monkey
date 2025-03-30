"""
Terminal User Interface for Monkey Document Analysis Toolkit
"""

from core.engine.logging import get_logger, error, warning, info, debug, trace

def run_tui(command_processor):
    """
    Run the Terminal User Interface with the given command processor

    Args:
        command_processor: The initialized CommandProcessor instance
    """
    logger = get_logger(__name__)

    try:
        # Import the actual TUI implementation
        from core.interface.tui_impl import MonkeyTUI

        # Create and run the TUI app
        app = MonkeyTUI(command_processor)
        app.run()
    except ImportError as e:
        error(f"Textual library not installed. {e}")
        error("Please install it with: pip install textual")
        error("Falling back to CLI mode.")
        command_processor.start()
    except Exception as e:
        error(f"Error initializing TUI: {str(e)}")
        import traceback
        trace(traceback.format_exc())
        error("Falling back to CLI mode.")
        command_processor.start()
