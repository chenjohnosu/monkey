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

    cli = CommandProcessor(config)

    if args.batch:
        if os.path.exists(args.batch):
            info(f"Running batch file: {args.batch}")
            config.set('system.batch_mode', True)

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            workspace = config.get('workspace.default', 'default')
            logs_dir = os.path.join('logs', workspace)
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            log_file = os.path.join(logs_dir, f"batch_{timestamp}.txt")

            # Set up logging to the file regardless of debug level
            LogManager.add_file_handler(log_file)
            info(f"Batch mode: logging output to {log_file}")

            # Redirect stdout/stderr in HPC mode, but ensure logging still works
            if args.hpc:
                # Save original stdout/stderr
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr

                # Open the log file for stdout/stderr redirection
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