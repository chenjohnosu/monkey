# Monkey Batch Mode

## Overview

Batch mode allows Monkey to execute a series of commands from a text file. This is particularly useful for automation, scheduled tasks, and running on HPC (High-Performance Computing) environments.

## Basic Usage

```bash
python monkey.py --batch /path/to/batch_file.txt
```

Or with the short option:

```bash
python monkey.py -b /path/to/batch_file.txt
```

## Batch File Format

A batch file is a simple text file containing one Monkey command per line. Commands should be written exactly as you would type them in the interactive CLI.

### Example Batch File

```
# This is a comment
/load ws research
/run grind
/run themes all
/run topics all
/run sentiment all
/save buffer
/quit
```

## Variables

You can define and use variables in batch files:

```
workspace = research_project
/load ws ${workspace}
output_format = json
/config output ${output_format}
/run themes all
```

Variables are defined with `name = value` syntax and referenced with `${name}` syntax.

## Flow Control

### Conditionals

You can use basic conditionals to check for file existence:

```
if exists data/myfile.txt then
/load ws project1
else
/load ws project2
endif
```

### Error Handling

By default, batch processing continues even if commands fail. You can change this behavior with the `EXIT_ON_ERROR` variable:

```
EXIT_ON_ERROR = true
/load ws research
/run nonexistent_command  # This will cause batch processing to stop
```

### Early Exit

You can exit batch processing early:

```
/load ws research
/run grind
exit 0  # Exit with success code
```

Or with an error code:

```
/run themes
if exists error.log then
exit 1  # Exit with error code
endif
```

## Best Practices

1. Begin batch files with workspace loading and configuration
2. Include explicit `/quit` at the end of batch files to ensure clean exit
3. Use comments to document complex operations
4. Set `EXIT_ON_ERROR = true` for critical workflows
5. Save outputs with `/save buffer` for later inspection

## Example Use Cases

### Data Pipeline

```
# Process new documents
workspace = incoming_data
/load ws ${workspace}
/run grind
/run themes all
/save buffer
```

### Scheduled Analysis

```
# Weekly analysis script
today = 2023-05-01
/load ws weekly_analysis_${today}
/run update
/run topics all
/run sentiment all
/save buffer
```

### HPC Cluster Job

```
# HPC batch job
EXIT_ON_ERROR = true
/config llm mistral
/config embed multilingual-e5
/load ws large_corpus
/run grind
/run themes all
/save buffer
```