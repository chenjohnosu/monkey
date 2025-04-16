"""
Terminal User Interface for Monkey Document Analysis Toolkit
"""

import logging
from core.engine.logging import error, warning, info, debug, trace
from core.engine.logging import LogManager

def run_tui(command_processor):
    """
    Run the Terminal User Interface with the given command processor

    Args:
        command_processor: The initialized CommandProcessor instance
    """
    logger = logging.getLogger(__name__)

    try:
        import textual
    except ImportError:
        warning("Textual library not installed. Falling back to CLI mode.")
        command_processor.start()
        return
    try:
        LogManager.set_tui_mode(True)
        from core.interface.tui_impl import MonkeyTUI
        app = MonkeyTUI(command_processor)
        app.run()
    except ImportError as e:
        LogManager.set_tui_mode(False)
        logger.error(f"TUI implementation error: {e}")
        logger.warning("Falling back to CLI mode.")
        command_processor.start()
    except Exception as e:
        LogManager.set_tui_mode(False)
        logger.error(f"Error initializing TUI: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        logger.warning("Falling back to CLI mode.")
        command_processor.start()