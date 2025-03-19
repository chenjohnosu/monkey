


Working Directories
ws = workspace - mirrors corpus vdb/raw source in the data and body directories
./body/[ws] = raw source (.txt, .md, .pdf files)
./data/[ws] = vdb and meta data storage

New Organization
./          - home directory
./core      - core functionality/source code
./data/[ws] - data storage
./body/[ws] - raw documents
./logs      - monitoring and debug logs 

New Organization 
- ./
  - ./monkey.py   - main entry point, output
  - ./config.yaml <-- default configuration override (optional; use default)
  - ./guides.txt  - Tagged directives for LLM
  - ./stopwords_en.txt - User Defined [default + user defined]
  - ./stopwords_zh.txt - User Defined [default + user defined]
  - ./core
    - ./core/engine (cli, config, utility, cuda check)
    - ./core/themes.py (mode) <-- thematic_analysis.py
    - ./core/query.py (mode) <-- query_engine
    - ./core/grind.py (mode) <-- grind, file processing
    - ./core/output.py <-- saving, logging, loading
    - ./core/storage.py <-- vector store, merge
- ./data/<workspace>
- ./src/<source>

COMMAND HIERARCHY

System Level
/quit, /ext                         - exit monkey
/show status                        - Show all user and systemn settings
/show cuda                          - Check nvidia CUDA Status 
/show config                        - Show default and user defined configurations
/show ws                            - Show workspace details
/show files                         - Dump file list with full paths associated with workspace and current system setting

Run Modes
/run themes [all|nfm|net|key]       - Theme analysis [MODE]
/run query                          - Enter interactive mode [MODE]
/run grind <ws>                     - Create vdb from ws name [MODE]
/run merge <src ws> <dst ws>        - Merge ws vdb [MODE]
FUTURE: /run topic                  - Topic Analysis [MODE] (Stub only)
FUTURE: /run stats                  - Statistical Analysis [MODE] (Stub only)

File Operations
/load ws <workspace>                - identify corpus or raw documents to examine
/load scan <workspace>              - list new/updated src data in <workspace>
/load guide <guide>                 - load tagged GUIDE entry from guide.txt
/save start                         - Save session 
/save stop                          - Stop saving
/save buffer                        - Save output of last command

Global Run Modifiers (defaults: set in config.yaml)
/config llm <model>                 - Set LLM for text generation (default: mistral)
/config embed <embedding model>     - Set embedding model (default: mixbread)
/config kval <n>                    - Number of nodes to retrieve as result in chinese policy?
/config debug [on|off]              - Turn debugging information ON/OFF (default: off)
/config output [txt|md|json]        - Output format (dfault: txt)

Help Commands
/help [run|load|config]             - Help 

Aliases
/q  --> /quit
/c  --> /config
/l  --> /load
/r  --> /run
/s  --> /save
/h  --> /help

Interface command line:

[d[workspace][llm model]