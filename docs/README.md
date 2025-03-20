# Monkey - Next-Generation Document Analysis Toolkit

Monkey is a powerful command-line document analysis toolkit designed for researchers, data scientists, and knowledge workers. It provides a flexible framework for processing, analyzing, and querying document collections with a focus on multilingual support (especially English and Chinese).

## Features

- **Advanced Document Processing**: Ingest and process multiple document formats (PDF, DOCX, TXT, MD, HTML)
- **Multilingual Support**: Specialized handling for both English and Chinese content with language-aware tokenization
- **Vector Search**: Semantic document retrieval using state-of-the-art embedding models
- **Thematic Analysis**: Multiple methodologies to extract themes, topics, and key concepts
- **Interactive Query Mode**: Ask questions about your document collection using LLM-powered responses
- **Local Inference**: Uses Ollama for local LLM inference without API costs or privacy concerns
- **Hardware Optimization**: Automatic detection and utilization of CUDA/GPU or Apple Silicon MPS

## Architecture

Monkey combines the power of established document understanding libraries with a flexible, modular design:

- **LlamaIndex Integration**: For document processing and semantic search
- **Haystack Integration**: For flexible pipeline-based document workflows
- **Local Embedding Models**: Uses powerful multilingual models like E5, mixbread, and BGE
- **Workspace Isolation**: Keep different document collections and analysis separate

## Installation

### Prerequisites

- Python 3.9 or higher
- [Ollama](https://ollama.ai) for local LLM inference
- Optional: CUDA-compatible GPU for acceleration

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

4. Install Ollama and download at least one model:
   - Follow the [Ollama installation instructions](https://github.com/ollama/ollama)
   - Download a model, e.g., `ollama pull phi4-mini`

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
   /run grind my_documents
   ```

5. Analyze themes in the documents:
   ```
   /run themes
   ```

6. Query your documents:
   ```
   /run query
   What are the main themes discussed in the documents?
   ```

7. Exit the application:
   ```
   /quit
   ```

## Configuration

Monkey uses a `config.yaml` file for persistent configuration. You can modify settings directly in the file or using the `/config` commands.

Default configuration:
```yaml
embedding:
  default_model: multilingual-e5
hardware:
  device: auto
  use_cuda: auto
  use_mps: auto
llm:
  default_model: phi4-mini
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
```

## Commands Overview

Monkey provides an intuitive command-line interface. Some key commands include:

- `/load <workspace>` - Load a workspace
- `/run grind <workspace>` - Process documents in a workspace
- `/run themes [all|nfm|net|key]` - Analyze themes in documents
- `/run query` - Enter interactive query mode
- `/config llm <model>` - Set the LLM model for text generation
- `/config embed <model>` - Set the embedding model
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
├── data/<workspace>/       - vector databases and metadata
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
- Language-appropriate embedding models


## Acknowledgments

Monkey builds upon several excellent open-source projects:
- [LlamaIndex](https://www.llamaindex.ai/)
- [Haystack](https://haystack.deepset.ai/)
- [Ollama](https://ollama.ai/)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)