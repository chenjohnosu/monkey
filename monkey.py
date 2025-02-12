# monkey v 0.6 Redev
#     Johnny's academic research tool to ingest data/notes/articles and allow interactive query.
#
#     "If you give enough monkeys enough typewriters, they will eventually type out the
#     complete works of Shakespeare." -- Émile Borel (1913)
#
#     This is a general tool that will take a directory full of PDFs and docx
#     files, convert them to .txt files, then create a vector database for
#     RAG use with multiple LLMs.
#
#     Johnny Chen
#     College of Business
#     Oregon State University
#     Center for Marketing & Consumer Insights
#     chenjohn@oregonstate.edu
#     10/28/2024
#
# INSTALLATION: Requires local Ollama
#
# 4 Modes
# --grind               Generate embeddings/vector DB (first pass; required)
# --wrench              Query Vector Database
#       --do            Single question on Command line
#       * default       Interactive Q&A mode
# --pmode               Experimental: SmartDataFrame / Interactive pandasai
# --merge               Expand vdb by merging new source to target vdb store
#
# Helper tools:         chimp.py
#                       If PDF is image based, use tesseract OCR to extract text
#                       Run chimp separately on a directory of image based PDFs then
#                       copy into biz directory or grind first then merge
#
# Dev Log:
# 0.4 Moved index as query engine to CitationQueryEngine; configured "--kn"
# 0.5 Added merge mode of two organs to grow db without having to re-index/embed
#     More error checking and code cleanup
# 0.6 Start Re-dev and streamlining
#
# NEW: Add all types of text data available for txt, docx, and pdfs; will ingest
#      .md, .epub, and .mbox but no error checking!
#
#
#
import logging
from pathlib import Path
import sys
import os

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Absolute imports instead of relative imports
from config import MonkeyConfig
from cli import CLI
from file_processor import FileProcessor
from vector_store import VectorStore
from query_engine import QueryEngine


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('monkey.log'),
            logging.StreamHandler()
        ]
    )


def interactive_chat(query_engine, verbose):
    """Handle interactive chat mode."""
    print("\nEntering interactive chat mode. Type 'exit' or 'quit' to end the session.")
    print("=" * 50)

    while True:
        try:
            # Get user input
            query = input("\nYour question: ").strip()

            # Check for exit command
            if query.lower() in ['exit', 'quit']:
                print("\nExiting chat mode...")
                break

            if not query:
                continue

            # Process the query
            result = query_engine.process_query(query, verbose)

            # Print the response
            print("\nResponse:")
            print("=" * 50)
            print("\n".join(result['response']))
            print("\nElapsed time: {:.2f}s".format(result['elapsed_time']))

            # Print sources if available
            if result['sources']:
                print("\nSources:")
                for source in result['sources']:
                    print(f"Source {source['id']}: {source['file']} (Score: {source.get('score', 'N/A')})")
                    if verbose and 'text' in source:
                        print(f"Text: {source['text']}")

            print("=" * 50)

        except KeyboardInterrupt:
            print("\nExiting chat mode...")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")


def main():
    """Main entry point for the monkey application."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Parse command line arguments
    cli = CLI()
    args = cli.parse_args()
    if not args:
        return

    # Load configuration
    try:
        config = MonkeyConfig.from_yaml('config.yaml')
    except Exception:
        logger.info("No config.yaml found, using default configuration")
        config = MonkeyConfig()

    # Override config with command line arguments
    if args.biz:
        config.src_dir = args.biz
    if args.organ:
        config.vdb_dir = args.organ
    if args.see:
        config.llm_model = args.see
    if args.temp is not None:
        config.temperature = args.temp
    if args.knoodles is not None:
        config.k_retrieve = args.knoodles

    # Initialize settings
    config.initialize_settings()

    # Process files if in grind mode
    if args.grind:
        src_path = Path(config.src_dir)
        if not src_path.exists():
            logger.error(f"Source directory {src_path} does not exist")
            return

        processor = FileProcessor(config)
        processor.process_directory(src_path)

        vector_store = VectorStore(config)
        vector_store.create_vector_store(src_path)
        logger.info("Vector store created successfully")
        return

    # Handle merge mode
    if args.merge and args.organ:
        vector_store = VectorStore(config)
        vector_store.merge_vector_stores(
            Path(args.merge),
            Path(args.organ)
        )
        logger.info("Vector stores merged successfully")
        return

    # Handle query mode
    if args.wrench or args.do:
        vector_store = VectorStore(config)
        index = vector_store.load_vector_store()
        if not index:
            logger.error("Failed to load vector store")
            return

        query_engine = QueryEngine(config, index)

        if args.do:
            # Single query mode
            result = query_engine.process_query(args.do, args.verbose)
            print("\n".join(result['response']))
            print(f"\nElapsed time: {result['elapsed_time']:.2f}s")
            for source in result['sources']:
                print(f"Source {source['id']}: {source['file']}")
                if args.verbose and 'text' in source:
                    print(f"Text: {source['text']}")
        else:
            # Interactive chat mode
            interactive_chat(query_engine, args.verbose)


if __name__ == "__main__":
    main()