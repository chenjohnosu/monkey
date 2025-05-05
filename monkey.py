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

if sys.platform.startswith('win'):
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def setup_directories():
    dirs = ['data', 'body', 'logs', 'lexicon']
    for dir_name in dirs:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            info(f"Created directory: {dir_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Monkey - Document Analysis Toolkit")
    parser.add_argument('-c', '--config', help='Custom configuration file')
    parser.add_argument('--debug', choices=['off', 'error', 'warning', 'info', 'debug', 'trace'],
                        default=None, help='Set debug level')
    parser.add_argument('-b', '--batch', help='Run commands from batch file')
    parser.add_argument('--hpc', action='store_true', help='HPC mode: suppress interactive prompts')
    return parser.parse_args()


def main():
    args = parse_arguments()
    setup_directories()

    config_path = args.config if args.config else 'config.yaml'
    config = Config(config_path)

    if args.debug is not None:
        config.set('system.debug_level', args.debug)
        LogManager.set_level(args.debug)
        debug(f"Debug level set to: {args.debug}")

    if args.hpc:
        config.set('system.hpc_mode', True)
        info("HPC mode enabled: suppressing interactive prompts")

    # Handle log file redirection if specified
    if args.log_file:
        info(f"Redirecting all logs to: {args.log_file}")
        LogManager.redirect_all_logs(args.log_file)

    cli = CommandProcessor(config)

    if args.batch:
        if os.path.exists(args.batch):
            info(f"Running batch file: {args.batch}")
            config.set('system.batch_mode', True)

            # Set up batch mode logging - enable console output and log to file
            from core.engine.logging import LogManager
            LogManager.configure(batch_mode=True)
            info(f"Batch mode: logging to console and default log file")

            # Redirect stdout/stderr in HPC mode, but ensure logging still works
            if args.hpc:
                # Save original stdout/stderr
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr

                # Open the log file for stdout/stderr redirection
                log_file = LogManager.get_log_file_path()
                log_file_handle = open(log_file, 'a', encoding='utf-8')
                sys.stdout = log_file_handle
                sys.stderr = log_file_handle

                info("HPC mode: redirected stdout and stderr to log file")

            success = cli.process_batch_file(args.batch)

            # Restore stdout/stderr if redirected
            if args.hpc and 'log_file_handle' in locals():
                sys.stdout = orig_stdout
                sys.stderr = orig_stderr
                log_file_handle.close()

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
        cli.start()


if __name__ == "__main__":
    main()