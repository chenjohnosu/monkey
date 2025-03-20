# Monkey Command Reference

This document provides a comprehensive reference for all commands available in the Monkey document analysis toolkit.

## Command Hierarchy

### System Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/quit`, `/exit` | `/q` | Exit Monkey |
| `/show status` | | Show all user and system settings |
| `/show cuda` | | Check NVIDIA CUDA status |
| `/show config` | | Show default and user-defined configurations |
| `/show ws` | | Show workspace details |
| `/show files` | | List files in the current workspace |
| `/show guide` | | List available LLM directive guides |
| `/help` | `/h` | Display general help information |
| `/help [command]` | | Display help for a specific command |

### Run Mode Commands

| Command | Description |
|---------|-------------|
| `/run themes [all\|nfm\|net\|key\|lsa\|cluster]` | Run theme analysis with specified method |
| `/run query` | Enter interactive query mode |
| `/run grind <ws>` | Process files in workspace (create vector database) |
| `/run grind <ws> scan` | Scan for new/updated files without processing |
| `/run grind <ws> force` | Force reprocessing of all files |
| `/run merge <src_ws> <dst_ws>` | Merge source workspace into destination workspace |

### File Operations

| Command | Alias | Description |
|---------|-------|-------------|
| `/load <workspace>` | `/l <workspace>` | Load or create a workspace |
| `/load guide <guide>` | `/l guide <guide>` | Load a tagged directive from guides.txt |

### Output Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/save start` | `/s start` | Start saving session output |
| `/save stop` | `/s stop` | Stop saving session output |
| `/save buffer` | `/s buffer` | Save output of the last command |

### Configuration Commands

| Command | Alias | Description |
|---------|-------|-------------|
| `/config llm <model>` | `/c llm <model>` | Set the LLM model for text generation |
| `/config embed <model>` | `/c embed <model>` | Set the embedding model |
| `/config kval <n>` | `/c kval <n>` | Set the number of documents to retrieve in queries |
| `/config debug [on\|off]` | `/c debug [on\|off]` | Enable or disable debug mode |
| `/config output [txt\|md\|json]` | `/c output [format]` | Set output format |
| `/config storage <backend>` | `/c storage <backend>` | Set storage backend (llama_index, haystack, chroma) |
| `/config guide <guide>` | `/c guide <guide>` | Set active directive guide |

### Inspection Commands

| Command | Description |
|---------|-------------|
| `/inspect workspace [ws]` | Inspect workspace metadata and vector store |
| `/inspect documents [ws] [limit]` | Dump document content (default: 5 documents) |
| `/inspect vectorstore [ws]` | Dump vector store metadata |
| `/inspect query [ws] [text]` | Test query pipeline with optional query text |
| `/inspect rebuild [ws]` | Rebuild vector store from existing documents |
| `/inspect fix [ws]` | Fix common vector store issues |
| `/inspect migrate [ws]` | Fix inconsistent vector store naming |

## Detailed Command Descriptions

### System Commands

#### `/quit`, `/exit`, `/q`
Exit the Monkey application.

#### `/show status`
Display the current status of the system, including active workspace, LLM model, embedding model, and other settings.

#### `/show cuda`
Check and display NVIDIA CUDA status, including availability, device count, device name, and total memory.

#### `/show config`
Display the full current configuration from config.yaml.

#### `/show ws`
Show details about the current workspace and any other loaded workspaces, including document counts and language distribution.

#### `/show files`
List all files in the current workspace with their sizes and modification dates.

#### `/show guide`
List all available directive guides defined in guides.txt.

#### `/help`, `/h`
Display general help information with a list of available commands.

#### `/help [command]`
Display detailed help for a specific command category (run, load, config, etc.).

### Run Mode Commands

#### `/run themes [method]`
Run theme analysis on documents in the current workspace using the specified method:
- `all`: Run all analysis methods (default)
- `nfm`: Named entity analysis
- `net`: Content network analysis
- `key`: Keyword extraction
- `lsa`: Latent semantic analysis
- `cluster`: Document clustering

#### `/run query`
Enter interactive query mode for the current workspace. In this mode, any text entered will be treated as a query to the document collection. LLM will generate responses based on retrieved context.

To exit query mode, type `/exit` or `/quit`.

#### `/run grind <workspace>`
Process all files in the specified workspace:
1. Scan for new or modified files
2. Extract and preprocess text
3. Store document metadata
4. Create vector embeddings

Options:
- `scan`: Only scan for new/modified files without processing
- `force`: Reprocess all files regardless of their change status

#### `/run merge <src_ws> <dst_ws>`
Merge documents from source workspace into destination workspace. Documents with the same paths will be skipped to avoid duplication.

### File Operations

#### `/load <workspace>`, `/l <workspace>`
Load an existing workspace or create a new one if it doesn't exist. All subsequent operations will use this workspace unless changed.

#### `/load guide <guide>`, `/l guide <guide>`
Load a tagged directive from guides.txt to guide LLM behavior during analysis and queries.

### Output Commands

#### `/save start`, `/s start`
Start saving all command outputs and responses to a session file.

#### `/save stop`, `/s stop`
Stop saving session output and close the session file.

#### `/save buffer`, `/s buffer`
Save the output of the last command to a file.

### Configuration Commands

#### `/config llm <model>`, `/c llm <model>`
Set the LLM model to use for text generation. The model must be available in Ollama.

Common models:
- `phi4-mini`: Smaller, faster model
- `llama3`: Balanced performance
- `mistral`: Alternative high-quality model

#### `/config embed <model>`, `/c embed <model>`
Set the embedding model for vector search.

Available models:
- `multilingual-e5`: Supports multiple languages including Chinese
- `mixbread`: Optimized for mixed-language content
- `bge`: Alternative multilingual model

#### `/config kval <n>`, `/c kval <n>`
Set the number of documents to retrieve for each query (default: 5).

#### `/config debug [on|off]`, `/c debug [on|off]`
Enable or disable debug mode to show detailed operation logs.

#### `/config output [format]`, `/c output [format]`
Set the output format for saved files:
- `txt`: Plain text (default)
- `md`: Markdown
- `json`: JSON format for programmatic use

#### `/config storage <backend>`, `/c storage <backend>`
Set the vector database backend:
- `llama_index`: Default, good general performance
- `haystack`: Alternative with pipeline capabilities
- `chroma`: Lightweight option

#### `/config guide <guide>`, `/c guide <guide>`
Set the active directive guide for LLM interactions from guides.txt.

### Inspection Commands

#### `/inspect workspace [ws]`, `/inspect ws [ws]`
Inspect workspace metadata, document count, vector store status, and language distribution.

#### `/inspect documents [ws] [limit]`
Dump document content from the workspace, limited to the specified number of documents (default: 5).

#### `/inspect vectorstore [ws]`, `/inspect vdb [ws]`
Dump vector store metadata, including structure and stats.

#### `/inspect query [ws] [text]`
Test the query pipeline with an optional test query (default: "test").

#### `/inspect rebuild [ws]`
Rebuild the vector store from existing documents in the workspace.

#### `/inspect fix [ws]`
Attempt to fix common vector store issues, such as corrupted files or inconsistent structure.

#### `/inspect migrate [ws]`
Fix inconsistent vector store naming conventions.

## Query Mode

When in query mode (after running `/run query`), you can:
- Type natural language queries directly
- Use `/exit` or `/quit` to return to the command mode
- Continue to use system commands by prefixing with `/`

## Examples

### Basic Workflow

```
/load research_papers
/run grind research_papers
/run themes
/run query
What are the main topics discussed in the research papers?
/exit
/quit
```

### Configuration Workflow

```
/config llm llama3
/config embed multilingual-e5
/config kval 8
/show status
/run query
What are the policy implications mentioned in these documents?
/exit
```

### Inspection Workflow

```
/inspect workspace research_papers
/inspect documents research_papers 3
/inspect rebuild research_papers
/run query
/exit
```