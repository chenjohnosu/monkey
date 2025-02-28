# Monkey Research Tool v0.8

> "If you give enough monkeys enough typewriters, they will eventually type out the complete works of Shakespeare." -- Émile Borel (1913)

A research tool that takes a directory full of PDFs and DOCX files, processes them, and creates a vector database for RAG (Retrieval-Augmented Generation) with various LLMs through Ollama.

## 🔄 Recent Updates

**Version 0.8:**
- Fixed FAISS dimension compatibility issues
- Added explicit embedding dimension configuration
- Improved vector store creation with proper dimension detection
- Enhanced error handling for dimension mismatches
- Better diagnostics for debugging embedding issues

## 📋 Prerequisites

- Python 3.9+
- Ollama (for LLM access)
- [FAISS](https://github.com/facebookresearch/faiss) library:
  - For CPU: `pip install faiss-cpu`
  - For GPU: `pip install faiss-gpu` (requires CUDA)

## 🛠️ Installation

1. Clone this repository:
   ```
   git clone <repository-url>
   cd monkey-research-tool
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. [Optional] Install OCR dependencies for enhanced PDF processing:
   ```
   pip install pytesseract pdf2image Pillow
   ```
   (You'll also need Tesseract OCR and Poppler installed on your system)

## 🔨 Usage

### 1. Generate Vector Database (First Step)

Process a directory of documents and create embeddings:

```
python monkey.py --grind --biz /path/to/documents --organ my_vector_db
```

### 2. Query the Vector Database

#### Interactive Mode:
```
python monkey.py --wrench --organ my_vector_db
```

#### Single Query Mode:
```
python monkey.py --wrench --do "What is the main finding about X?" --organ my_vector_db
```

### 3. Advanced Options

- `--unique`: Retrieve k unique source files instead of potentially overlapping chunks
- `--see model_name`: Use specific LLM model (default: "mistral")
- `--temp 0.7`: Set temperature for LLM (default: 0.7)
- `--knoodles 5`: Number of sources to retrieve (default: 5)
- `--verbose`: Show detailed source information
- `--guide "Your custom guide text"`: Override default guide prompt for the LLM

### 4. Merge Vector Stores

Combine two vector stores:
```
python monkey.py --merge source_vdb --organ target_vdb
```

### 5. Topic Modeling

Perform topic modeling on the document collection:
```
python monkey.py --topics --topic-words 10 --organ my_vector_db
```

## 🔍 Troubleshooting FAISS Dimension Issues

If you encounter FAISS dimension mismatch errors, here are some solutions:

1. **Rebuild the vector store**: If your embedding model has changed or you're experiencing dimension errors, completely rebuild the vector store:
   ```
   python monkey.py --grind --biz /path/to/documents --organ new_vector_db
   ```

2. **Check config.yaml**: The tool now explicitly tracks embedding dimensions. You can set this in config.yaml:
   ```yaml
   embedding_dimension: 1024
   embedding_model: "mixedbread-ai/mxbai-embed-large-v1"
   ```

3. **Dimension Detection**: The tool will try to automatically detect the correct embedding dimension from your model. If you switch embedding models, make sure to rebuild your vector store.

## 📄 Configuration

Create a `config.yaml` file for persistent configuration:

```yaml
src_dir: "src"
vdb_dir: "vdb"
llm_model: "mistral"
temperature: 0.7
k_retrieve: 5
line_width: 80
chunk_size: 1024
chunk_overlap: 200
embedding_model: "mixedbread-ai/mxbai-embed-large-v1" 
embedding_dimension: 1024
guide: "You are a very intelligent text wrangler and researcher."
```

## 📊 Performance Notes

- **GPU Acceleration**: If CUDA is available, the tool will use GPU acceleration for embeddings and FAISS operations.
- **Memory Usage**: For large document collections, consider increasing batch size or using a machine with more RAM.
- **FAISS Optimization**: The tool now correctly handles FAISS dimension configuration for better performance.

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve this tool.

## 📝 License

This project is licensed under the terms of the MIT license.