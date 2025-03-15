import gc
import logging
import time
import torch
from typing import List, Optional, Dict, Any
from pathlib import Path
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    Document
)

from config import MonkeyConfig
from file_processor import FileProcessor
from cuda_utils import CUDAChecker


class VectorStore:
    def __init__(self, config: MonkeyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.index = None
        self.file_processor = FileProcessor(config)
        self.cuda_checker = CUDAChecker()

    def _monitor_cuda_usage(self) -> Dict[str, Any]:
        """Monitor CUDA memory usage and statistics."""
        if not torch.cuda.is_available():
            return {
                "available": False,
                "memory_allocated": 0,
                "memory_reserved": 0,
                "max_memory_allocated": 0
            }

        return {
            "available": True,
            "memory_allocated": torch.cuda.memory_allocated() / 1024 ** 2,  # MB
            "memory_reserved": torch.cuda.memory_reserved() / 1024 ** 2,  # MB
            "max_memory_allocated": torch.cuda.max_memory_allocated() / 1024 ** 2  # MB
        }

    def create_vector_store(self, input_dir: Path, batch_size: int = 10) -> None:
        """Create vector store directly from documents in batches."""
        self.logger.info(f"Creating vector store from {input_dir}")
        start_time = time.time()

        # Display initial CUDA status
        cuda_info = self.cuda_checker.check_cuda_availability()
        print("\nCUDA Status for Vector Store Creation:")
        print(f"CUDA Available: {'Yes' if cuda_info['cuda_available'] else 'No'}")
        if cuda_info['cuda_available']:
            print(f"GPU Device: {cuda_info['device_name']}")
            print(f"CUDA Version: {cuda_info['cuda_version']}")
            cuda_usage = self._monitor_cuda_usage()
            print(f"Initial GPU Memory Usage: {cuda_usage['memory_allocated']:.2f} MB")
            print(f"GPU Memory Reserved: {cuda_usage['memory_reserved']:.2f} MB")
        print(f"Embedding Device: {self.cuda_checker.check_embedding_device()}")

        # Process all files in directory and get Document objects
        all_documents = self.file_processor.process_directory(input_dir)
        total_batches = (len(all_documents) + batch_size - 1) // batch_size

        #print(f"\nProcessing {len(all_documents)} documents in {total_batches} batches")

        # Process documents in batches
        for i in range(0, len(all_documents), batch_size):
            batch = all_documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            # print(f"\nProcessing batch {batch_num}/{total_batches}")
            self._process_batch(batch)
            gc.collect()  # Clean up memory after each batch

            # Show progress statistics
            elapsed_time = time.time() - start_time
            docs_processed = min(i + batch_size, len(all_documents))
            avg_time_per_doc = elapsed_time / docs_processed

            #print(f"Progress Statistics:")
            #print(f"Documents Processed: {docs_processed}/{len(all_documents)}")
            #print(f"Average Time per Document: {avg_time_per_doc:.2f}s")
            #print(f"Total Time Elapsed: {elapsed_time:.2f}s")

            # Monitor CUDA usage after batch
            #if cuda_info['cuda_available']:
            #    cuda_usage = self._monitor_cuda_usage()
            #    print(f"\nCurrent GPU Memory Usage:")
            #    print(f"Allocated: {cuda_usage['memory_allocated']:.2f} MB")
            #    print(f"Reserved: {cuda_usage['memory_reserved']:.2f} MB")
            #    print(f"Max Used: {cuda_usage['max_memory_allocated']:.2f} MB")

        if self.index:
            #print("\nPersisting final vector store...")
            # Monitor CUDA usage before persistence
            #if cuda_info['cuda_available']:
            #    pre_persist_cuda = self._monitor_cuda_usage()
            #    print(f"\nGPU Memory Before Persistence:")
            #    print(f"Allocated: {pre_persist_cuda['memory_allocated']:.2f} MB")
            #    print(f"Reserved: {pre_persist_cuda['memory_reserved']:.2f} MB")
            #    print(f"Max Used: {pre_persist_cuda['max_memory_allocated']:.2f} MB")

            # Persist the vector store
            self.index.storage_context.persist(persist_dir=self.config.vdb_dir)

            # Monitor CUDA usage after persistence
            #if cuda_info['cuda_available']:
            #    post_persist_cuda = self._monitor_cuda_usage()
            #    print(f"\nGPU Memory After Persistence:")
            #    print(f"Allocated: {post_persist_cuda['memory_allocated']:.2f} MB")
            #    print(f"Reserved: {post_persist_cuda['memory_reserved']:.2f} MB")
            #    print(f"Max Used: {post_persist_cuda['max_memory_allocated']:.2f} MB")

            #   # Calculate memory changes
            #   memory_change = post_persist_cuda['memory_allocated'] - pre_persist_cuda['memory_allocated']
            #   print(f"Memory Change During Persistence: {memory_change:.2f} MB")

            final_time = time.time() - start_time
            print(f"\nVector Store Creation Complete:")
            print(f"Total Processing Time: {final_time:.2f}s")
            print(f"Average Time per Document: {final_time / len(all_documents):.2f}s")
            print(f"Documents Processed: {len(all_documents)}")
            print(f"Vector Store Location: {self.config.vdb_dir}")

            # Final CUDA utilization summary
            #if cuda_info['cuda_available']:
            #    print(f"\nFinal CUDA Utilization Summary:")
            #    print(f"Peak Memory Usage: {post_persist_cuda['max_memory_allocated']:.2f} MB")
            #    print(f"Final Memory Allocated: {post_persist_cuda['memory_allocated']:.2f} MB")
            #    print(
            #        f"Memory Efficiency: {(post_persist_cuda['memory_allocated'] / post_persist_cuda['memory_reserved'] * 100):.1f}%")

    def _process_batch(self, documents: List[Document]) -> None:
        """Process a batch of Document objects."""
        try:
            batch_start_time = time.time()
            #print("\nAdding batch to vector store...")

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
                    show_progress=False,
                    use_async=True
                )
            else:
                self.index.insert_nodes(documents)

            #batch_time = time.time() - batch_start_time
            #print(
            #    f"✓ Successfully processed batch in {batch_time:.2f}s"
            #    f"\n  - Added {len(documents)} document sections from {len(file_sections)} files"
            #    f"\n  - Current embedding device: {self.cuda_checker.check_embedding_device()}"
            #)

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