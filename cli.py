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
            description='Research RAG for pdf, txt, and docx (and data).'
        )
        parser.add_argument('-b', '--biz', type=str,
                          help='Directory of documents: (default: src)')
        parser.add_argument('-d', '--do', type=str,
                          help='Single Query to the documents')
        parser.add_argument('-g', '--grind', action="store_true",
                          help='MODE: Create vector store with documents')
        parser.add_argument('-k', '--knoodles', type=int,
                          help='Manually configure k-retrieved (default: 5)')
        parser.add_argument('-m', '--merge', type=str,
                          help='MODE: Merge identified vdb to main organ')
        parser.add_argument('-o', '--organ', type=str,
                          help='Vector store name (default: vdb)')
        parser.add_argument('-s', '--see', type=str,
                          help='Use LLM Model (default: mistral)')
        parser.add_argument('-t', '--temp', type=float,
                          help='Set LLM temperature (default: 0.7)')
        parser.add_argument('-w', '--wrench', action="store_true",
                          help='MODE: Load vector store to query')
        parser.add_argument('-v', '--verbose', action="store_true",
                          help='Show sources and detailed debug information')
        return parser

    def parse_args(self) -> Optional[argparse.Namespace]:
        """Parse command line arguments."""
        args = self.parser.parse_args()
        
        if not len(sys.argv) > 1:
            self.parser.print_help()
            return None

        if args.grind and args.wrench:
            print("Cannot use both grind and wrench modes simultaneously")
            return None

        return args
