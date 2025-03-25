# Monkey - Next-Generation Document Analysis Toolkit

Monkey is a powerful command-line document analysis toolkit designed for researchers, data scientists, and knowledge workers. It provides a flexible framework for processing, analyzing, and querying document collections with a focus on multilingual support (especially English and Chinese).

## Features

- **Advanced Document Processing**: Ingest and process multiple document formats (PDF, DOCX, TXT, MD, HTML)
- **Multilingual Support**: Specialized handling for both English and Chinese content with language-aware tokenization
- **Vector Search**: Semantic document retrieval using state-of-the-art embedding models
- **Thematic Analysis**: Multiple methodologies to extract themes, topics, and key concepts
- **Topic Modeling**: Analyze document collections using LDA, NMF, and clustering techniques
- **Sentiment Analysis**: Evaluate emotional content in documents with specialized support for Chinese text
- **Interactive Query Mode**: Ask questions about your document collection using LLM-powered responses
- **LLM-Assisted Interpretation**: Get AI-powered interpretations of analysis results
- **Local Inference**: Uses Ollama for local LLM inference without API costs or privacy concerns
- **Hardware Optimization**: Automatic detection and utilization of CUDA/GPU or Apple Silicon MPS

## Architecture

Monkey combines the power of established document understanding libraries with a flexible, modular design:

- **LlamaIndex Integration**: For document processing and semantic search
- **Haystack Integration**: For flexible pipeline-based document workflows
- **Local Embedding Models**: Uses powerful multilingual models like E5, mixbread, BGE, and Jina for Chinese optimization
- **Workspace Isolation**: Keep different document collections and analysis separate
- **Modular Connector Design**: Easily swap components to customize your workflow

## Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai) for local LLM inference
- Optional: CUDA-compatible GPU for acceleration
- Optional: Apple Silicon Mac for MPS acceleration

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/monkey.git
   cd monkey
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Optional (recommended for Chinese text): Install jieba
   ```bash
   pip install jieba
   ```

5. Install Ollama and download at least one model:
   - Follow the [Ollama installation instructions](https://github.com/ollama/ollama)
   - Download a model, e.g., `ollama pull mistral` or `ollama pull phi4-mini`

## Getting Started

1. Start the Monkey interface:
   ```bash
   python monkey.py
   ```

2. Create and load a workspace:
   ```
   /load my_documents
   ```

3. Add documents to your workspace by placing them in the `body/my_documents/` directory

4. Process the documents:
   ```
   /run grind
   ```

5. Analyze themes in the documents:
   ```
   /run themes
   ```

6. Run topic modeling:
   ```
   /run topic
   ```

7. Generate sentiment analysis:
   ```
   /run sentiment
   ```

8. Get AI interpretation of your results:
   ```
   /explain themes What are the most significant patterns across themes?
   ```

9. Query your documents:
   ```
   /run query
   What are the main themes discussed in the documents?
   ```

10. Exit the application:
    ```
    /quit
    ```

## Configuration

Monkey uses a `config.yaml` file for persistent configuration. You can modify settings directly in the file or using the `/config` commands.

Default configuration:
```yaml
embedding:
  default_model: multilingual-e5
  chinese_model: jina-zh
hardware:
  device: auto
  use_cuda: auto
  use_mps: auto
llm:
  default_model: mistral
  ollama_host: http://localhost
  ollama_port: 11434
  source: ollama
query:
  k_value: 5
storage:
  vector_store: llama_index
system:
  debug: false
  output_format: txt
workspace:
  default: default
topic:
  use_originals: true
```

## Commands Overview

Monkey provides an intuitive command-line interface. Some key commands include:

- `/load <workspace>` - Load a workspace
- `/run grind` - Process documents in the current workspace
- `/run update` - Update workspace with new or modified files
- `/run themes [all|nfm|net|key|lsa|cluster]` - Analyze themes using different methods
- `/run topic [all|lda|nmf|cluster]` - Run topic modeling
- `/run sentiment [all|basic|advanced]` - Perform sentiment analysis
- `/run query` - Enter interactive query mode
- `/explain themes|topics|sentiment [question]` - Get LLM interpretation of analysis
- `/config llm <model>` - Set the LLM model for text generation
- `/config embed <model>` - Set the embedding model
- `/clear [logs|vdb|cache|all]` - Clear logs, vector databases, or cached data
- `/inspect [workspace|documents|vectorstore|query|rebuild|fix|migrate]` - Inspect and fix issues
- `/help` - Show available commands

For a comprehensive list of commands, see the [COMMAND.md](COMMAND.md) file.

## Data Organization

```
./                          - home directory
├── monkey.py               - main entry point
├── config.yaml             - configuration file
├── guides.txt              - LLM directive guides
├── stopwords_en.txt        - English stopwords
├── stopwords_zh.txt        - Chinese stopwords
├── core/                   - core source code
│   ├── connectors/         - component connectors (LlamaIndex, Haystack, Ollama)
│   ├── engine/             - core processing engine
│   ├── language/           - language processing modules
│   └── modes/              - analysis mode implementations
├── data/<workspace>/       - vector databases and metadata
│   ├── documents/          - processed document data
│   └── vector_store/       - vector embeddings and indexes
├── logs/<workspace>/       - analysis output logs
└── body/<workspace>/       - raw document files
```

## Advanced Usage

### Customizing Stopwords

For better analysis results, you can customize the stopwords files:
- `stopwords_en.txt` - English stopwords list
- `stopwords_zh.txt` - Chinese stopwords list

### Using Directive Guides

The `guides.txt` file contains tagged directives for different analysis scenarios. You can set these with:
```
/config guide <guide_name>
```

Available guides include `chinese_policy`, `technical_analysis`, `academic_research`, and others.

### Working with Chinese Documents

Monkey has specialized support for Chinese document analysis:
- Automatic language detection
- Character-based tokenization (with jieba if available)
- Chinese stopword removal
- Specialized embedding models (jina-zh optimized for Chinese)
- Configure Chinese-specific embedding:
  ```
  /config embed jina-zh
  ```

### Maintenance and Optimization

Maintaining your workspaces for optimal performance:
- Clear logs when no longer needed: `/clear logs`
- Rebuild vector stores if searching issues arise: `/inspect rebuild`
- Fix common vector database issues: `/inspect fix`
- Migrate vector stores with naming inconsistencies: `/inspect migrate`

## Acknowledgments

Monkey builds upon several excellent open-source projects:
- [LlamaIndex](https://www.llamaindex.ai/)
- [Haystack](https://haystack.deepset.ai/)
- [Ollama](https://ollama.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Jieba](https://github.com/fxsjy/jieba) for Chinese word segmentation