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
# 5 Modes
# --grind               Generate embeddings/vector DB (first pass; required)
# --wrench              Query Vector Database
#       --do            Single question on Command line
#       * default       Interactive Q&A mode
# --pmode               Experimental: SmartDataFrame / Interactive pandasai
# --merge               Expand vdb by merging new source to target vdb store
# --topics              Untrained Topic Modeling
#
#
# Dev Log:
# 0.4 Moved index as query engine to CitationQueryEngine; configured "--kn"
# 0.5 Added merge mode of two organs to grow db without having to re-index/embed
#     More error checking and code cleanup
# 0.61 Start Re-dev and streamlining; modular design; topic modeling; guide
#      command line
# 0.7 CUDA optimizations, diagnostics, and monitoring
# NEW: Add all types of text data available for txt, docx, and pdfs; will ingest
#      .md, .epub, and .mbox but no error checking!
#
#
#



import logging
from pathlib import Path
import sys
import os
import traceback

# Add the parent directory to sys.path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MonkeyConfig
from cli import CLI
from file_processor import FileProcessor
from vector_store import VectorStore
from query_engine import QueryEngine


def setup_logging():
    """Setup logging configuration with log rotation."""
    from logging.handlers import RotatingFileHandler
    import logging
    import sys

    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Set up rotating file handler
    log_file = 'monkey.log'
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024 * 1024,  # 1MB per file
        backupCount=5,         # Keep 5 backup files
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Set up console handler for critical errors only
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.CRITICAL)
    console_handler.setFormatter(formatter)

    # Remove any existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add the new handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Log uncaught exceptions
    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception

    # Log startup message
    logger.info("Logging system initialized")

def display_system_info(config: MonkeyConfig):
    """Display system configuration and diagnostic information."""
    print("\nSystem Configuration:")
    print("=" * 50)
    print(f"Vector Store: {config.vdb_dir}")
    print(f"Language Model: {config.llm_model}")
    print(f"Temperature: {config.temperature}")
    print(f"k-retrieve: {config.k_retrieve}")
    print(f"Chunk Size: {config.chunk_size}")
    print(f"Chunk Overlap: {config.chunk_overlap}")
    print(f"Embedding Model: {config.embedding_model}")
    print(f"Guide: {config.guide}")
    print("=" * 50)


def display_response_diagnostics(result: dict, config: MonkeyConfig):
    """Display diagnostic information for a response."""
    print("\nResponse Diagnostics:")
    print("-" * 50)
    print(f"Model: {config.llm_model}")
    print(f"Temperature: {config.temperature}")
    print(f"Response Time: {result['elapsed_time']:.2f}s")
    print(f"Sources Retrieved: {len(result['sources'])}")

    if result['sources']:
        print("\nSource Details:")
        for source in result['sources']:
            print(f"[{source['id']}] {source['file']} | Score: {source['score']}")



def interactive_chat(query_engine, verbose, config):
    """Handle interactive chat mode with diagnostic information."""
    print("\nEntering interactive chat mode. Type 'exit' or 'quit' to end the session.")
    display_system_info(config)

    while True:
        try:
            query = input("\nYour question: ").strip()

            if query.lower() in ['exit', 'quit']:
                print("\nExiting chat mode...")
                break

            if not query:
                continue

            result = query_engine.process_query(query, verbose=verbose)

            print("\nResponse:")
            print("=" * 50)
            print("\n".join(result['response']))

            display_response_diagnostics(result, config)

            if verbose and result['sources']:
                print("\nSource Content:")
                for source in result['sources']:
                    if 'text' in source:
                        print(f"\n[{source['id']}] Text:")
                        print(source['text'])

            print("=" * 50)

        except KeyboardInterrupt:
            print("\nExiting chat mode...")
            break
        except Exception as e:
            print(f"\nError processing query: {str(e)}")


def main():
    """Main entry point for the monkey application."""
    # Set up logging first thing
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info("Starting monkey application")

        # Parse command line arguments
        cli = CLI()
        args = cli.parse_args()
        if not args:
            logger.info("No arguments provided, exiting")
            return

        logger.info(f"Parsed arguments: {args}")

        # Load configuration
        try:
            config = MonkeyConfig.from_yaml('config.yaml')
            logger.info("Loaded configuration from config.yaml")
        except Exception as e:
            logger.info(f"No config.yaml found or error loading it: {e}, using default configuration")
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
        if args.guide is not None:
            config.guide = args.guide

        logger.info(f"Final configuration: {vars(config)}")

        # Initialize settings
        config.initialize_settings()
        logger.info("Initialized settings")

        # Process files if in grind mode
        if args.grind:
            logger.info("Starting grind mode")
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
            logger.info("Starting merge mode")
            vector_store = VectorStore(config)
            vector_store.merge_vector_stores(
                Path(args.merge),
                Path(args.organ)
            )
            logger.info("Vector stores merged successfully")
            return

        # Handle query mode
        if args.wrench or args.do:
            logger.info("Starting query mode")
            vector_store = VectorStore(config)
            index = vector_store.load_vector_store()
            if not index:
                logger.error("Failed to load vector store")
                return

            query_engine = QueryEngine(config, index)

            if args.do:
                # Single query mode
                logger.info(f"Processing single query: {args.do}")
                result = query_engine.process_query(args.do, verbose=args.verbose)
                print("\n".join(result['response']))
                display_response_diagnostics(result, config)
                if args.verbose and result['sources']:
                    print("\nSource Content:")
                    for source in result['sources']:
                        if 'text' in source:
                            print(f"\n[{source['id']}] Text:")
                            print(source['text'])
            else:
                # Interactive chat mode
                interactive_chat(query_engine, args.verbose, config)

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise  # Re-raise the exception after logging it


if __name__ == "__main__":
    main()