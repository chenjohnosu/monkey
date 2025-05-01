#!/usr/bin/env python3
"""
Monkey - Next-generation document analysis toolkit
Main entry point for the application
"""

import os
import sys
import argparse
import datetime
from core.engine.cli import CommandProcessor
from core.engine.config import Config
from core.engine.logging import LogManager, error, warning, info, debug

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
    parser.add_argument('-c', '--config', help='Custom configuration file')
    parser.add_argument('--debug', choices=['off', 'error', 'warning', 'info', 'debug', 'trace'],
                        default=None, help='Set debug level')
    parser.add_argument('-b', '--batch', help='Run commands from batch file')
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

    # Set debug level if specified
    if args.debug is not None:
        config.set('system.debug_level', args.debug)
        LogManager.set_level(args.debug)
        debug(f"Debug level set to: {args.debug}")

    # Set HPC mode if specified
    if args.hpc:
        config.set('system.hpc_mode', True)
        info("HPC mode enabled: suppressing interactive prompts")

    # Initialize command processor
    cli = CommandProcessor(config)

    # Execute in batch mode if batch file is specified
    if args.batch:
        if os.path.exists(args.batch):
            info(f"Running batch file: {args.batch}")

            # Set up batch mode in config
            config.set('system.batch_mode', True)

            # Set up logging to a file in the logs directory
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            workspace = config.get('workspace.default', 'default')
            logs_dir = os.path.join('logs', workspace)
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            log_file = os.path.join(logs_dir, f"batch_{timestamp}.txt")
            LogManager.add_file_handler(log_file)
            info(f"Batch mode: logging output to {log_file}")

            # Redirect stdout to the batch log file if in HPC mode
            if args.hpc:
                sys.stdout = open(log_file, 'w', encoding='utf-8')
                sys.stderr = open(log_file, 'a', encoding='utf-8')
                info("HPC mode: redirected stdout and stderr to log file")

            # Process the batch file
            success = cli.process_batch_file(args.batch)

            # Restore stdout/stderr if redirected
            if args.hpc and sys.stdout != sys.__stdout__:
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

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