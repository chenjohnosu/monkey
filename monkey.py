#!/usr/bin/env python3
"""
Monkey - Next-generation document analysis toolkit
Main entry point for the application
"""

import os
import sys
import argparse
from core.engine.cli import CommandProcessor
from core.engine.config import Config
from core.engine.logging import error, warning, info, trace, debug, debug

# Fix for Windows event loop policy
if sys.platform.startswith('win'):
    import asyncio

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
    parser.add_argument('-b', '--batch', help='Run commands from batch file')
    parser.add_argument('-q', '--query', help='Execute a one-time query against the workspace')
    parser.add_argument('-r', '--run', help='Run a specific command (format: "command arg1 arg2...")')
    parser.add_argument('--hpc', action='store_true', help='HPC mode: suppress interactive prompts')
    return parser.parse_args()


def main():
    """Main entry point for the application"""
    # Parse command line arguments
    args = parse_arguments()

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

    # Set HPC mode if specified
    if args.hpc:
        config.set('system.hpc_mode', True)
        info("HPC mode enabled: suppressing interactive prompts")

    # Initialize command processor
    cli = CommandProcessor(config)

    # Load initial workspace if specified
    if args.workspace:
        cli.process_command(f"/load ws {args.workspace}")

    # Execute command if specified
    if args.run:
        info(f"Running command: {args.run}")
        cli.process_command(f"/{args.run}")
        sys.exit(0)

    # Execute one-time query if specified
    if args.query:
        if not args.workspace:
            error("Workspace must be specified with -w when using query mode")
            sys.exit(1)

        info(f"Executing one-time query: {args.query}")
        cli.process_command(f"/run query {args.query}")
        sys.exit(0)

    # Execute in batch mode if batch file is specified
    if args.batch:
        if os.path.exists(args.batch):
            info(f"Running batch file: {args.batch}")
            success = cli.process_batch_file(args.batch)
            if not success:
                error(f"Failed to process batch file: {args.batch}")
                sys.exit(1)
            else:
                info(f"Batch processing complete: {args.batch}")
                sys.exit(0)
        else:
            error(f"Batch file not found: {args.batch}")
            sys.exit(1)
    else:
        # Start interactive command processing loop
        cli.start()


if __name__ == "__main__":
    main()