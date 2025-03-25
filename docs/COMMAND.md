# Monkey Command Reference Guide

This guide provides a comprehensive overview of all commands available in the Monkey document analysis toolkit. Each command is explained with examples and usage notes.

## Command Basics

Monkey uses a command-line interface with a consistent syntax:

```
/<command> [subcommand] [arguments]
```

For example:
```
/run themes all
```

In this example:
- `/run` is the command
- `themes` is the subcommand
- `all` is an argument

## Command Categories

Monkey commands are organized into these categories:

1. **Run Commands**: Execute analysis and processing operations
2. **Load Commands**: Load workspaces and resources
3. **Config Commands**: Configure system settings
4. **Show Commands**: Display information and status
5. **Save Commands**: Save analysis results and sessions
6. **Inspect Commands**: Diagnose and fix issues
7. **Explain Commands**: Get LLM interpretations of results
8. **Clear Commands**: Clean up logs and databases
9. **Utility Commands**: Help, exit, etc.

## 1. Run Commands

### `/run grind [workspace]`

Process documents in a workspace to create the initial database.

```
/run grind                # Process current workspace
/run grind my_workspace   # Process specific workspace
```

### `/run update [workspace]`

Update workspace with new or modified files.

```
/run update               # Update current workspace
```

### `/run scan [detailed]`

Scan workspace for new or updated files without processing them.

```
/run scan                 # Basic scan of current workspace
/run scan detailed        # Show detailed file information
```

### `/run themes [method]`

Run theme analysis on documents with the specified method.

```
/run themes               # Run all theme analysis methods
/run themes nfm           # Use named entity-focused method
/run themes net           # Use content network method
/run themes key           # Use keyword extraction method
/run themes lsa           # Use latent semantic analysis
/run themes cluster       # Use document clustering
```

### `/run topic [method]`

Run topic modeling on documents with the specified method.

```
/run topic                # Run all topic modeling methods
/run topic lda            # Use Latent Dirichlet Allocation
/run topic nmf            # Use Non-Negative Matrix Factorization
/run topic cluster        # Use clustering-based topic modeling
```

### `/run sentiment [method]`

Run sentiment analysis on documents with the specified method.

```
/run sentiment            # Run all sentiment analysis methods
/run sentiment basic      # Use basic lexicon-based analysis
/run sentiment advanced   # Use advanced analysis with aspect extraction
```

### `/run query`

Enter interactive query mode for the current workspace.

```
/run query                # Start interactive query mode
```

In query mode, type `/exit` or `/quit` to return to the main command mode.

### `/run merge <source_workspace>`

Merge source workspace into the current workspace.

```
/run merge old_docs       # Merge old_docs into current workspace
```

## 2. Load Commands

### `/load <workspace>`

Load or create a workspace.

```
/load my_workspace        # Load or create 'my_workspace'
```

If the workspace doesn't exist, you'll be prompted to confirm creation.

## 3. Config Commands

### `/config llm <model>`

Set the LLM model for text generation.

```
/config llm mistral       # Set LLM model to mistral
/config llm phi4-mini     # Set LLM model to phi4-mini
```

### `/config embed <model>`

Set the embedding model for document vectorization.

```
/config embed multilingual-e5   # General multilingual model
/config embed jina-zh           # Optimized for Chinese
/config embed mixbread          # Alternative embedding model
/config embed bge               # Another embedding option
```

### `/config kval <n>`

Set the k value for retrieval (number of documents to retrieve for each query).

```
/config kval 5            # Set k value to 5
/config kval 10           # Set k value to 10
```

### `/config debug [on|off]`

Set debug mode.

```
/config debug on          # Enable debug mode
/config debug off         # Disable debug mode
```

### `/config storage <backend>`

Set storage backend for vector database.

```
/config storage llama_index  # Use LlamaIndex backend
/config storage haystack     # Use Haystack backend
/config storage chroma       # Use Chroma backend
```

### `/config output [txt|md|json]`

Set output format for saved results.

```
/config output txt        # Plain text format
/config output md         # Markdown format
/config output json       # JSON format
```

### `/config guide <guide>`

Set active guide from guides.txt.

```
/config guide chinese_policy      # Set Chinese policy analysis guide
/config guide technical_analysis  # Set technical analysis guide
```

## 4. Show Commands

### `/show status`

Show system status including active workspace, models, and configuration.

```
/show status
```

### `/show cuda`

Show CUDA status and GPU availability.

```
/show cuda
```

### `/show config`

Show detailed configuration.

```
/show config
```

### `/show ws`

Show workspace details and statistics.

```
/show ws
```

### `/show files`

Show files in current workspace.

```
/show files
```

### `/show guide`

Show available guides in guides.txt.

```
/show guide
```

## 5. Save Commands

### `/save start`

Start saving session output to a file.

```
/save start
```

### `/save stop`

Stop saving session output.

```
/save stop
```

### `/save buffer`

Save last command output buffer to a file.

```
/save buffer
```

## 6. Inspect Commands

### `/inspect workspace [workspace]`

Inspect workspace metadata and vector store.

```
/inspect workspace           # Inspect current workspace
/inspect workspace old_docs  # Inspect specific workspace
/inspect ws                  # Shorthand for workspace inspect
```

### `/inspect documents [workspace] [limit]`

Dump document content in workspace.

```
/inspect documents           # Show first 5 documents in current workspace
/inspect documents old_docs  # Show first 5 documents in specific workspace
/inspect documents 10        # Show first 10 documents
```

### `/inspect vectorstore [workspace]`

Dump vector store metadata.

```
/inspect vectorstore         # Show vector store details for current workspace
/inspect vdb                 # Shorthand for vectorstore
```

### `/inspect query [workspace] [query]`

Test query pipeline with an optional test query.

```
/inspect query               # Test with default query
/inspect query "test query"  # Test with specific query
```

### `/inspect rebuild [workspace]`

Rebuild vector store from existing documents.

```
/inspect rebuild             # Rebuild vector store for current workspace
```

### `/inspect fix [workspace]`

Fix common vector store issues.

```
/inspect fix                 # Fix issues in current workspace
```

### `/inspect migrate [workspace]`

Fix inconsistent vector store naming.

```
/inspect migrate             # Fix naming in current workspace
```

### `/inspect metadata [workspace] [query] [limit]`

Inspect raw metadata returned from vector store.

```
/inspect metadata            # Show metadata for default query
/inspect metadata "test" 5   # Show metadata for "test" query, 5 results
```

## 7. Explain Commands

### `/explain themes [question]`

Get LLM interpretation of theme analysis results.

```
/explain themes              # General theme interpretation
/explain themes What themes are most significant?  # Specific question
```

### `/explain topics [question]`

Get LLM interpretation of topic modeling results.

```
/explain topics              # General topic interpretation
/explain topics How do the topics relate to each other?  # Specific question
```

### `/explain sentiment [question]`

Get LLM interpretation of sentiment analysis results.

```
/explain sentiment           # General sentiment interpretation
/explain sentiment What emotions are most prominent?  # Specific question
```

### `/explain session [question]`

Get LLM interpretation of query session.

```
/explain session             # General session interpretation
/explain session What were the main research directions?  # Specific question
```

## 8. Clear Commands

### `/clear logs [workspace]`

Clear log files for a workspace.

```
/clear logs                  # Clear logs for current workspace
```

### `/clear vdb [workspace]`

Clear vector database files for a workspace.

```
/clear vdb                   # Clear vector DB for current workspace
```

### `/clear cache [workspace]`

Clear cached data and intermediary files.

```
/clear cache                 # Clear cache for current workspace
```

### `/clear all [workspace]`

Clear all logs, vector database, and cache files.

```
/clear all                   # Clear everything for current workspace
```

## 9. Utility Commands

### `/help [topic]`

Display help information.

```
/help                        # General help
/help run                    # Help for run commands
/help config                 # Help for config commands
/help clear                  # Help for clear commands
```

### `/quit` or `/exit`

Exit the application (or exit query mode if in query mode).

```
/quit                        # Exit application
/exit                        # Same as /quit
```

## Command Aliases

Monkey provides short aliases for frequently used commands:

| Alias | Full Command |
|-------|-------------|
| `/q` | `/quit` |
| `/c` | `/config` |
| `/l` | `/load` |
| `/r` | `/run` |
| `/s` | `/save` |
| `/h` | `/help` |
| `/i` | `/inspect` |
| `/e` | `/explain` |
| `/cl` | `/clear` |

## Query Mode Commands

When in query mode (`/run query`), you can use these special commands:

- `/exit` or `/quit` - Exit query mode and return to command mode
- Any other text is treated as a query to the document collection

## Best Practices

1. **Start with workspace setup**:
   ```
   /load my_workspace
   ```

2. **Process your documents**:
   ```
   /run grind
   ```

3. **Check document stats**:
   ```
   /show ws
   ```

4. **Run analyses in sequence**:
   ```
   /run themes
   /run topic
   /run sentiment
   ```

5. **Get AI interpretation**:
   ```
   /explain themes
   ```

6. **Query your documents**:
   ```
   /run query
   ```

7. **Regularly maintain your workspace**:
   ```
   /clear logs
   /inspect fix
   ```

8. **Save important sessions**:
   ```
   /save start
   # ... do important work ...
   /save stop
   ```

By using these commands effectively, you can extract valuable insights from your document collections with the Monkey toolkit.