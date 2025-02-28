import gc
import logging
import time
import torch
import faiss
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

    def _create_faiss_index(self, dimension: int = None):
        """Create a FAISS index with the correct dimension."""
        # Use the config's dimension or default to 1024
        if dimension is None:
            dimension = getattr(self.config, 'embedding_dimension', 1024)

        print(f"Creating FAISS index with dimension: {dimension}")

        # Create a new FAISS index with the specified dimension
        faiss_index = faiss.IndexFlatL2(dimension)

        # Use GPU if available
        if torch.cuda.is_available():
            try:
                res = faiss.StandardGpuResources()
                faiss_index = faiss.index_cpu_to_gpu(res, 0, faiss_index)
                print("✓ FAISS index moved to GPU")
            except Exception as e:
                print(f"Error moving FAISS index to GPU: {str(e)}")
                print("× Will use CPU FAISS index instead")

        # Return the raw FAISS index as LlamaIndex will handle it appropriately
        return faiss_index

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

        if len(all_documents) == 0:
            print("No documents found to process!")
            return

        # Generate a test embedding to determine the dimension
        try:
            test_embedding = self.config.initialize_settings()
            test_text = "Test document for dimension detection"
            test_embedding = self.config.Settings.embed_model.get_text_embedding(test_text)
            detected_dimension = len(test_embedding)
            print(f"Detected embedding dimension: {detected_dimension}")
            self.config.embedding_dimension = detected_dimension
        except Exception as e:
            print(f"Error detecting embedding dimension: {str(e)}")
            print(f"Using default dimension: {self.config.embedding_dimension}")

        # Create FAISS index with the correct dimension
        faiss_index = self._create_faiss_index(self.config.embedding_dimension)
        # Use LlamaIndex to handle the vector store creation with the FAISS index
        storage_context = StorageContext.from_defaults(vector_store_kwargs={"faiss_index": faiss_index})

        print(f"\nProcessing {len(all_documents)} documents in {total_batches} batches")

        # Process first batch to initialize index
        first_batch = all_documents[:min(batch_size, len(all_documents))]
        try:
            print("\nInitializing vector store with first batch...")
            self.index = VectorStoreIndex.from_documents(
                first_batch,
                storage_context=storage_context,
                show_progress=True,
                use_async=True
            )
            print("✓ Vector store initialized successfully")
        except Exception as e:
            print(f"✗ Error initializing vector store: {str(e)}")
            self.logger.error(f"Error initializing vector store: {str(e)}")
            return

        # Process remaining documents in batches
        remaining_docs = all_documents[min(batch_size, len(all_documents)):]
        for i in range(0, len(remaining_docs), batch_size):
            batch = remaining_docs[i:i + batch_size]
            batch_num = (i // batch_size) + 2  # +2 because we already processed one batch
            print(f"\nProcessing batch {batch_num}/{total_batches}")
            self._process_batch(batch)
            gc.collect()  # Clean up memory after each batch

            # Show progress statistics
            elapsed_time = time.time() - start_time
            docs_processed = min(i + batch_size + batch_size, len(all_documents))
            avg_time_per_doc = elapsed_time / docs_processed

            print(f"Progress Statistics:")
            print(f"Documents Processed: {docs_processed}/{len(all_documents)}")
            print(f"Average Time per Document: {avg_time_per_doc:.2f}s")
            print(f"Total Time Elapsed: {elapsed_time:.2f}s")

        if self.index:
            print("\nPersisting final vector store...")
            # Persist the vector store
            self.index.storage_context.persist(persist_dir=self.config.vdb_dir)

            final_time = time.time() - start_time
            print(f"\nVector Store Creation Complete:")
            print(f"Total Processing Time: {final_time:.2f}s")
            print(f"Average Time per Document: {final_time / len(all_documents):.2f}s")
            print(f"Documents Processed: {len(all_documents)}")
            print(f"Vector Store Location: {self.config.vdb_dir}")
            print(f"Embedding Dimension: {self.config.embedding_dimension}")

    def _process_batch(self, documents: List[Document]) -> None:
        """Process a batch of Document objects."""
        try:
            batch_start_time = time.time()

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

            # Insert documents into the existing index
            self.index.insert_nodes(documents)

            batch_time = time.time() - batch_start_time
            print(
                f"✓ Successfully processed batch in {batch_time:.2f}s"
                f"\n  - Added {len(documents)} document sections from {len(file_sections)} files"
                f"\n  - Using embedding dimension: {self.config.embedding_dimension}"
            )

        except Exception as e:
            print(f"✗ Error processing batch: {str(e)}")
            self.logger.error(f"Error processing batch: {str(e)}")

    def load_vector_store(self) -> Optional[VectorStoreIndex]:
        """Load existing vector store."""
        try:
            print(f"Loading vector store from {self.config.vdb_dir}...")

            storage_context = StorageContext.from_defaults(
                persist_dir=self.config.vdb_dir
            )

            self.index = load_index_from_storage(storage_context)

            # Check if the index loaded correctly
            if self.index is None:
                raise ValueError("Failed to load index from storage")

            print(f"✓ Vector store loaded successfully")

            # Try to infer the embedding dimension
            if hasattr(self.index.vector_store, "client"):
                if hasattr(self.index.vector_store.client, "d"):
                    detected_dimension = self.index.vector_store.client.d
                    print(f"Detected FAISS index dimension: {detected_dimension}")
                    self.config.embedding_dimension = detected_dimension

            return self.index

        except Exception as e:
            print(f"✗ Error loading vector store: {str(e)}")
            self.logger.error(f"Error loading vector store: {str(e)}")
            print("If this is due to dimension mismatch, try rebuilding the vector store with --grind")
            return None

    def merge_vector_stores(self, source_dir: Path, target_dir: Path) -> None:
        """Merge two vector stores."""
        try:
            print(f"Merging vector stores: {source_dir} -> {target_dir}")

            # Load source vector store
            print(f"Loading source vector store: {source_dir}")
            source_storage_context = StorageContext.from_defaults(persist_dir=str(source_dir))
            source_index = load_index_from_storage(source_storage_context)

            # Load target vector store
            print(f"Loading target vector store: {target_dir}")
            target_storage_context = StorageContext.from_defaults(persist_dir=str(target_dir))
            target_index = load_index_from_storage(target_storage_context)

            # Get dimension information
            source_dimension = None
            target_dimension = None

            # Try to infer dimensions from vector stores
            if hasattr(source_index.vector_store, "client") and hasattr(source_index.vector_store.client, "d"):
                source_dimension = source_index.vector_store.client.d
                print(f"Source vector store dimension: {source_dimension}")

            if hasattr(target_index.vector_store, "client") and hasattr(target_index.vector_store.client, "d"):
                target_dimension = target_index.vector_store.client.d
                print(f"Target vector store dimension: {target_dimension}")

            # Check if dimensions match
            if source_dimension is not None and target_dimension is not None and source_dimension != target_dimension:
                print(
                    f"WARNING: Source dimension ({source_dimension}) and target dimension ({target_dimension}) do not match!")
                print("Merging may fail. Consider rebuilding both vector stores with the same embedding model.")

            # Prepare nodes and embeddings for merging
            nodes = []

            # Get nodes and embeddings from source
            print("Collecting nodes from source vector store...")
            source_vector_store_dict = source_index.vector_store.to_dict()
            source_embedding_dict = source_vector_store_dict.get('embedding_dict', {})

            for doc_id, node in source_index.docstore.docs.items():
                if doc_id in source_embedding_dict:
                    node.embedding = source_embedding_dict[doc_id]
                    nodes.append(node)

            print(f"Collected {len(nodes)} nodes from source vector store")

            # Get nodes and embeddings from target
            print("Collecting nodes from target vector store...")
            target_vector_store_dict = target_index.vector_store.to_dict()
            target_embedding_dict = target_vector_store_dict.get('embedding_dict', {})

            target_nodes_count = 0
            for doc_id, node in target_index.docstore.docs.items():
                if doc_id in target_embedding_dict:
                    node.embedding = target_embedding_dict[doc_id]
                    nodes.append(node)
                    target_nodes_count += 1

            print(f"Collected {target_nodes_count} nodes from target vector store")
            print(f"Total nodes for merged vector store: {len(nodes)}")

            # Create a new FAISS index with the correct dimension
            dimension = source_dimension or target_dimension or self.config.embedding_dimension
            faiss_index = self._create_faiss_index(dimension)
            # Use LlamaIndex to handle the vector store creation with the FAISS index
            storage_context = StorageContext.from_defaults(vector_store_kwargs={"faiss_index": faiss_index})

            # Create merged index
            print("Creating merged vector store...")
            merged_index = VectorStoreIndex(nodes=nodes, storage_context=storage_context)

            # Persist merged index
            merged_dir = f"{target_dir}_merged"
            print(f"Persisting merged vector store to: {merged_dir}")
            merged_index.storage_context.persist(persist_dir=merged_dir)

            print(f"✓ Vector stores successfully merged to: {merged_dir}")

        except Exception as e:
            print(f"✗ Error merging vector stores: {str(e)}")
            self.logger.error(f"Error merging vector stores: {str(e)}")