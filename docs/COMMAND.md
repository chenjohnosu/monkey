# Monkey Data Loading and Formatting Guide

This guide provides best practices for loading, organizing, and formatting data for optimal analysis with the Monkey document analysis toolkit.

## Directory Structure

Monkey uses a specific directory structure to organize document data:

```
./                          - home directory
├── body/<workspace>/       - raw document files
└── data/<workspace>/       - vector databases and metadata
```

- **body/**: Contains the original document files organized by workspace
- **data/**: Stores processed data, including vector embeddings and metadata

## Creating Workspaces

Workspaces isolate different document collections for separate analysis:

1. Create a workspace directory:
   ```
   /load my_workspace
   ```

2. Add documents to the workspace directory:
   ```
   cp my_documents/*.pdf body/my_workspace/
   ```

3. Process the documents:
   ```
   /run grind my_workspace
   ```

## Supported File Formats

Monkey supports the following file formats:

| Format | Extension | Notes |
|--------|-----------|-------|
| Text | `.txt` | Plain text, any encoding (UTF-8 preferred) |
| Markdown | `.md` | Preserves basic formatting |
| PDF | `.pdf` | Requires PyPDF2 or pdfminer.six library |
| Word | `.docx` | Requires python-docx library |
| HTML | `.html` | Basic tag removal |

## Document Organization Strategies

### Flat Organization

For smaller document collections, a flat organization works well:

```
body/my_workspace/
├── document1.pdf
├── document2.docx
├── document3.txt
└── ...
```

### Hierarchical Organization

For larger collections, organize documents in subdirectories:

```
body/my_workspace/
├── category1/
│   ├── doc1.pdf
│   └── doc2.pdf
├── category2/
│   ├── doc3.docx
│   └── doc4.docx
└── ...
```

Monkey preserves directory structure in document metadata, allowing for filtering by path during analysis.

## Document Preparation Tips

### General Tips

1. **Clean your documents**: Remove headers, footers, and other boilerplate content when possible
2. **Use descriptive filenames**: Filenames are preserved in metadata and can help with analysis
3. **Ensure proper encoding**: Use UTF-8 encoding for best multilingual support
4. **Remove password protection**: Encrypted files cannot be processed

### Tips for English Documents

1. **Remove stopwords**: Consider removing common stopwords before processing
2. **Normalize case**: Consider converting all text to lowercase
3. **Remove special characters**: Remove special characters that don't contribute to meaning

### Tips for Chinese Documents

1. **Character encoding**: Ensure UTF-8 encoding for Chinese character support
2. **Word segmentation**: Install jieba for better Chinese word segmentation:
   ```
   pip install jieba
   ```
3. **Custom stopwords**: Add domain-specific Chinese stopwords to `stopwords_zh.txt`

## Optimizing PDF Documents

PDF documents can be challenging to process. For best results:

1. Ensure PDFs are text-based, not scanned images
2. Remove any document security features
3. Consider pre-extracting text from complex PDFs
4. For multilingual PDFs, ensure proper font embedding

## Customizing Stopwords

Monkey uses stopword files to improve analysis quality:

- `stopwords_en.txt`: English stopwords
- `stopwords_zh.txt`: Chinese stopwords

To customize stopwords:

1. Edit the respective file
2. Add one stopword per line
3. For Chinese, include full characters
4. Reprocess the documents with `/run grind <workspace> force`

Example English stopwords:
```
a
about
above
after
again
...
```

Example Chinese stopwords:
```
的
了
和
与
这个
...
```

## Working with Large Document Collections

For large document collections:

1. **Split into multiple workspaces**: Divide by topic, date, or other logical groupings
2. **Process incrementally**: Add documents in batches
3. **Monitor memory usage**: Adjust batch sizes if memory issues occur
4. **Use a GPU**: Enable CUDA for faster processing

Process large collections incrementally:
```
# First batch
cp batch1/*.pdf body/my_workspace/
/run grind my_workspace

# Next batch
cp batch2/*.pdf body/my_workspace/
/run grind my_workspace
```

## Document Metadata

Monkey extracts and stores metadata for each document:

| Metadata | Description |
|----------|-------------|
| `source` | Relative path to the original document |
| `workspace` | Workspace name |
| `language` | Detected language (e.g., 'en', 'zh') |
| `tokens` | Number of tokens in the document |
| `content_hash` | Hash of document content for change detection |
| `last_modified` | Last modification timestamp |
| `file_size` | Document size in bytes |
| `processed_date` | When the document was processed |

## Multilingual Workflow

For multilingual document collections:

1. Use the `multilingual-e5` embedding model:
   ```
   /config embed multilingual-e5
   ```

2. Ensure proper language detection:
   - English documents should be primarily English
   - Chinese documents should contain sufficient Chinese characters for detection

3. Process documents:
   ```
   /run grind my_workspace
   ```

4. Verify language detection:
   ```
   /inspect workspace my_workspace
   ```

## Merging Workspaces

To combine document collections:

```
/run merge source_workspace destination_workspace
```

This preserves unique documents and avoids duplicates.

## Optimizing for Theme Analysis

For better theme analysis results:

1. Group related documents together
2. Use descriptive file paths that reflect content categories
3. Remove very short documents that don't contribute meaningful content
4. Remove duplicate or near-duplicate documents
5. For mixed-language collections, split by language into separate workspaces

## Using Different Vector Stores

Monkey supports multiple vector store backends:

```
/config storage llama_index   # Default, good general performance
/config storage haystack      # Alternative with pipeline capabilities
/config storage chroma        # Lightweight option
```

After changing the storage backend, rebuild the vector store:
```
/inspect rebuild my_workspace
```

## Monitoring and Troubleshooting

### Checking Document Processing

Monitor document processing:
```
/run grind my_workspace scan
```

This shows which documents need processing without actually processing them.

### Fixing Vector Store Issues

If you encounter issues with vector searches:
```
/inspect fix my_workspace
```

For more serious issues, rebuild the vector store:
```
/inspect rebuild my_workspace
```

### Examining Document Content

Check if documents were processed correctly:
```
/inspect documents my_workspace 3
```

This displays the content of the first 3 documents.

## Best Practices Summary

1. **Organize logically**: Create workspaces that group related documents
2. **Clean input data**: Ensure documents are properly formatted and encoded
3. **Monitor processing**: Check language detection and document counts
4. **Customize stopwords**: Add domain-specific stopwords for better analysis
5. **Process incrementally**: Add and process documents in manageable batches
6. **Test queries**: Verify search quality with test queries
7. **Use appropriate embedding models**: Select models based on your language needs

By following these guidelines, you'll get the best analysis results from the Monkey document analysis toolkit.