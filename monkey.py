#!/usr/bin/env python3
"""
Monkey - Next-generation document analysis toolkit
Main entry point for the application
"""

# Add this environment variable setting at the very top
# This disables tokenizers parallelism to prevent the fork warnings
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import argparse
import asyncio  # Import asyncio
from core.engine.cli import CommandProcessor
from core.engine.config import Config
from core.engine.logging import error, warning, info, trace, debug, debug

# Fix for Windows event loop policy
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def setup_directories():
    """Create necessary directories if they don't exist"""
    dirs = ['data', 'body', 'logs', 'lexicon']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Created directory: {dir_name}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Monkey - Document Analysis Toolkit")
    parser.add_argument('-w', '--workspace', help='Initial workspace to load')
    parser.add_argument('-c', '--config', help='Custom configuration file')
    parser.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--tui', action='store_true', help='Use Terminal User Interface')
    return parser.parse_args()


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()

    # Check for TUI mode flag and set it immediately
    if args.tui:
        from core.engine.logging import LogManager
        LogManager.set_tui_mode(True)

    # Set up necessary directories
    setup_directories()

    # Load configuration
    config_path = args.config if args.config else 'config.yaml'
    config = Config(config_path)

    # Override debug setting if specified in arguments
    if args.debug:
        config.set('system.debug', True)
        # Update global debug flag
        from core.engine.logging import LogManager
        LogManager.set_debug(True)
        debug("Debug mode enabled")

    # Initialize command processor
    cli = CommandProcessor(config)

    # Load initial workspace if specified
    if args.workspace:
        cli.process_command(f"/load ws {args.workspace}")

    # Start the appropriate interface
    if args.tui:
        try:
            # Import TUI module
            from core.interface.tui import run_tui
            # Run TUI interface
            info("Starting Terminal User Interface...")
            run_tui(cli)
        except ImportError:
            warning("TUI module not found. Falling back to CLI mode.")
            cli.start()
        except Exception as e:
            error(f"Error starting TUI: {str(e)}")
            info("Falling back to CLI mode.")
            cli.start()
    else:
        # Start command processing loop in CLI mode
        cli.start()


if __name__ == "__main__":
    main()