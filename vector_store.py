import gc
import logging
from typing import List, Optional
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)

from config import MonkeyConfig
from file_processor import FileProcessor


class VectorStore:
    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.file_processor = FileProcessor(config)

    def create_vector_store(self, input_dir: Path, batch_size: int = 10) -> None:
        """Create vector store directly from documents in batches."""
        self.logger.info(f"Creating vector store from {input_dir}")

        # Process all files in directory and get Document objects
        all_documents = self.file_processor.process_directory(input_dir)

        # Process documents in batches
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            self._process_batch(batch)
            gc.collect()  # Clean up memory after each batch

        if self.index:
            self.logger.info("Persisting final vector store...")
            self.index.storage_context.persist(persist_dir=self.config.vdb_dir)

    def _process_batch(self, documents: List[Document]) -> None:
        """Process a batch of Document objects."""
        try:
            print("\nAdding batch to vector store...")

            # Group documents by source file
            file_sections = {}
            for doc in documents:
                file_name = doc.metadata.get('file_name', 'Unknown file')
                if file_name not in file_sections:
                    file_sections[file_name] = 0
                file_sections[file_name] += 1

            # Print summary by file
            for file_name, section_count in file_sections.items():
                print(f"Adding to vector store: {file_name} ({section_count} sections)")

            if not self.index:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    show_progress=True
                )
            else:
                self.index.insert_nodes(documents)

            print(
                f"✓ Successfully added {len(documents)} document sections from {len(file_sections)} files to vector store")

        except Exception as e:
            print(f"✗ Error processing batch: {str(e)}")
            self.logger.error(f"Error processing batch: {str(e)}")

    def load_vector_store(self) -> Optional[VectorStoreIndex]:
        """Load existing vector store."""
        try:
            storage_context = StorageContext.from_defaults(
                persist_dir=self.config.vdb_dir
            )
            self.index = load_index_from_storage(storage_context)
            return self.index
        except Exception as e:
            self.logger.error(f"Error loading vector store: {str(e)}")
            return None

    def merge_vector_stores(self, source_dir: Path, target_dir: Path) -> None:
        """Merge two vector stores."""
        try:
            nodes = []
            for path in [source_dir, target_dir]:
                storage_context = StorageContext.from_defaults(persist_dir=str(path))
                index = load_index_from_storage(storage_context)

                vector_store_dict = index.storage_context.vector_store.to_dict()
                embedding_dict = vector_store_dict['embedding_dict']

                for doc_id, node in index.storage_context.docstore.docs.items():
                    node.embedding = embedding_dict[doc_id]
                    nodes.append(node)

            merged_index = VectorStoreIndex(nodes=nodes)
            merged_index.storage_context.persist(
                persist_dir=str(target_dir) + "_merged"
            )

        except Exception as e:
            self.logger.error(f"Error merging vector stores: {str(e)}")