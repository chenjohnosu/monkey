import gc
import logging
from typing import List, Optional
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage
)

# Import MonkeyConfig
from config import MonkeyConfig

class VectorStore:
    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.index = None

    def create_vector_store(self, input_dir: Path, batch_size: int = 1000) -> None:
        """Create vector store from documents in batches."""
        self.logger.info(f"Creating vector store from {input_dir}")
        
        documents = []
        for file_path in input_dir.rglob('*.txt'):
            documents.append(str(file_path))

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            self._process_batch(batch)
            gc.collect()

    def _process_batch(self, document_paths: List[str]) -> None:
        """Process a batch of documents."""
        try:
            documents = SimpleDirectoryReader(
                input_files=document_paths,
                recursive=True,
                required_exts=[".txt", ".md", ".mbox", ".epub"]
            ).load_data()
            
            if not self.index:
                self.index = VectorStoreIndex.from_documents(documents, show_progress=True)
            else:
                self.index.insert_nodes(documents)
                
            self.index.storage_context.persist(persist_dir=self.config.vdb_dir)
            
        except Exception as e:
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

