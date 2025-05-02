#!/usr/bin/env python3

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
            print(f"Created directory: {dir_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Monkey - Document Analysis Toolkit")
    parser.add_argument('-c', '--config', help='Custom configuration file')
    parser.add_argument('--debug', choices=['off', 'error', 'warning', 'info', 'debug', 'trace'],
                        default=None, help='Set debug level')
    parser.add_argument('-b', '--batch', help='Run commands from batch file')
    parser.add_argument('--hpc', action='store_true', help='HPC mode: optimize for high-performance computing')
    parser.add_argument('--cmd', help='Run a single command and exit')
    parser.add_argument('-w', '--workspace', help='Set the workspace to use')
    parser.add_argument('-o', '--output', help='Specify output file for results')
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

    if args.workspace:
        config.set('workspace.default', args.workspace)
        info(f"Using workspace: {args.workspace}")

    if args.hpc:
        config.set('system.hpc_mode', True)
        info("HPC mode enabled: optimizing for high-performance computing")

    cli = CommandProcessor(config)

    # First handle batch mode
    if args.batch:
        if os.path.exists(args.batch):
            info(f"Running batch file: {args.batch}")
            config.set('system.batch_mode', True)

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            workspace = config.get('workspace.default', 'default')
            logs_dir = os.path.join('logs', workspace)
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)

            # Determine log file, either from args or default
            log_file = args.output if args.output else os.path.join(logs_dir, f"batch_{timestamp}.txt")
            LogManager.add_file_handler(log_file)
            info(f"Batch mode: logging output to {log_file}")

            # Redirect stdout/stderr in HPC mode
            if args.hpc:
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr
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

    # Then handle single command mode
    elif args.cmd:
        info(f"Running single command: {args.cmd}")
        config.set('system.batch_mode', True)

        # Set up logging to file if output specified
        if args.output:
            LogManager.add_file_handler(args.output)
            info(f"Logging output to {args.output}")

            # Redirect stdout/stderr in HPC mode
            if args.hpc:
                orig_stdout = sys.stdout
                orig_stderr = sys.stderr
                log_file_handle = open(args.output, 'a', encoding='utf-8')
                sys.stdout = log_file_handle
                sys.stderr = log_file_handle

        # Execute the command
        cli.process_command(args.cmd)

        # Restore stdout/stderr if redirected
        if args.hpc and args.output and 'log_file_handle' in locals():
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file_handle.close()

        sys.exit(0)

    # Finally, fall back to interactive mode
    else:
        cli.run_interactive()


if __name__ == "__main__":
    main()