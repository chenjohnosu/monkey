import argparse
import sys
import logging
from pathlib import Path
from typing import Optional


class CLI:
    def __init__(self):
        self.parser = self._create_parser()

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            prog="monkey",
            description='Research LLM harness for pdf, txt, and docx (and data).'
        )
        parser.add_argument('-b', '--biz', type=str,
                            help='Directory of documents: (default: src)')
        parser.add_argument('-d', '--do', type=str,
                            help='Single Query to the documents')
        parser.add_argument('-g', '--grind', action="store_true",
                            help='MODE: Create vector store with documents')
        parser.add_argument('-k', '--knoodles', type=int,
                            help='Number of sources to retrieve (default: 5)')
        parser.add_argument('-m', '--merge', type=str,
                            help='MODE: Merge identified vdb to main organ')
        parser.add_argument('-o', '--organ', type=str,
                            help='Vector store name (default: vdb)')
        parser.add_argument('-s', '--see', type=str,
                            help='Use LLM Model (default: mistral)')
        parser.add_argument('-t', '--temp', type=float,
                            help='Set LLM temperature (default: 0.7)')
        parser.add_argument('-u', '--unique', action="store_true",
                            help='Retrieve k unique source files instead of potentially overlapping chunks')
        parser.add_argument('-w', '--wrench', action="store_true",
                            help='MODE: Load vector store to query')
        parser.add_argument('--topics', action="store_true",
                            help='MODE: Perform automatic topic modeling on vector store')
        parser.add_argument('--topic-words', type=int, default=10,
                            help='Number of words to show per topic (default: 10)')

        # New arguments for thematic analysis
        parser.add_argument('--themes', action="store_true",
                            help='MODE: Perform comprehensive thematic analysis across all documents')
        parser.add_argument('--theme-method', type=str, choices=['all', 'nmf', 'network', 'phrases'],
                            default='all', help='Thematic analysis method to use (default: all)')
        parser.add_argument('--min-occurrence', type=int, default=3,
                            help='Minimum occurrence threshold for concepts (default: 3)')
        parser.add_argument('--output-format', type=str, choices=['txt', 'json'],
                            default='txt', help='Output format for thematic analysis (default: txt)')
        parser.add_argument('--save-results', action="store_true",
                            help='Save thematic analysis results to files')

        parser.add_argument('-v', '--verbose', action="store_true",
                            help='Show sources and detailed debug information')
        parser.add_argument('--guide', type=str,
                            help='Override default guide prompt for the LLM')
        return parser

    def parse_args(self) -> Optional[argparse.Namespace]:
        """Parse command line arguments."""
        args = self.parser.parse_args()

        if not len(sys.argv) > 1:
            self.parser.print_help()
            return None

        # Check for mutually exclusive modes
        modes = [args.grind, args.wrench, args.merge is not None, args.topics, args.themes]
        if sum(modes) > 1:
            print("Error: Can only use one mode at a time (grind, wrench, merge, topics, or themes)")
            return None

        return args