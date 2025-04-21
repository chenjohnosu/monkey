"""
Terminal User Interface for Monkey Document Analysis Toolkit
"""

import os
import sys
import asyncio
from core.engine.logging import error, warning, info, debug, trace
from core.engine.logging import LogManager

def run_tui(command_processor):
    """
    Run the Terminal User Interface with the given command processor

    Args:
        command_processor: The initialized CommandProcessor instance
    """
    # Fix for Windows event loop policy - apply before any other asyncio operations
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Ensure we have a fresh event loop
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    except Exception as e:
        error(f"Failed to create event loop: {e}")

    # Set TUI mode for logging
    LogManager.set_tui_mode(True)

    try:
        # Import the actual TUI implementation
        from core.interface.tui_impl import MonkeyTUI

        # Create TUI app
        app = MonkeyTUI(command_processor)

        # Signal that TUI is ready
        LogManager.set_tui_ready(True)

        # Start the app with proper exception handling
        try:
            app.run()
        except KeyboardInterrupt:
            # Handle clean Ctrl+C exit
            info("Received keyboard interrupt. Exiting...")
        except Exception as e:
            error(f"Error in TUI application: {str(e)}")
            import traceback
            trace(traceback.format_exc())
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
    finally:
        # Clean up event loop
        try:
            loop = asyncio.get_event_loop()
            loop.close()
        except Exception as e:
            debug(f"Error closing event loop: {e}")