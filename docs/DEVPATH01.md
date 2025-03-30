# Monkey Document Analysis Toolkit: Production Readiness Implementation Plan

## 1. Dependency Management

### Issues
- No explicit dependency management
- Conditional imports with fallbacks
- Missing required vs. optional dependency distinction

### Implementation Plan
1. **Create a formal dependency specification**
   ```bash
   # Create requirements.txt
   pip freeze > requirements.txt
   
   # Clean up and organize into categories
   # core: required dependencies
   # gpu: CUDA dependencies
   # ui: TUI dependencies
   # full: all dependencies
   ```

2. **Implement proper package structure**
   ```bash
   # Create setup.py for installable package
   touch setup.py
   
   # Define entry points
   python -m pip install -e .
   ```

3. **Add dependency checking on startup**
   ```python
   def check_dependencies():
       """Check for required dependencies and report missing ones"""
       required = {
           "llama_index": "LlamaIndex for vector storage",
           "haystack": "Haystack for pipeline components",
           # Add other core dependencies
       }
       
       missing = []
       for package, description in required.items():
           try:
               __import__(package)
           except ImportError:
               missing.append(f"{package} ({description})")
       
       if missing:
           print("Missing required dependencies:")
           for dep in missing:
               print(f"  - {dep}")
           print("Install with: pip install -r requirements.txt")
           sys.exit(1)
   ```

## 2. Code Quality and Consistency

### Issues
- Inconsistent formatting and style
- Variable type hinting
- Mixed error handling approaches

### Implementation Plan
1. **Apply code formatters**
   ```bash
   # Install formatters
   pip install black isort mypy pylint
   
   # Format all Python files
   black core/ --line-length=100
   isort core/ --profile black
   
   # Add pre-commit hooks
   pip install pre-commit
   pre-commit install
   ```

2. **Add type hints consistently**
   ```bash
   # Run mypy to identify missing types
   mypy core/
   
   # Systematically add type hints to:
   # - Function parameters
   # - Return values
   # - Class attributes
   ```

3. **Standardize error handling**
   ```python
   # Create centralized error handling
   class MonkeyError(Exception):
       """Base exception for all Monkey errors"""
       pass
       
   class ConfigError(MonkeyError):
       """Configuration related errors"""
       pass
       
   # Add similar specialized exceptions
   ```

## 3. Testing Infrastructure

### Issues
- No visible unit tests
- Lack of test coverage
- Manual testing approach

### Implementation Plan
1. **Set up testing framework**
   ```bash
   # Install pytest and related tools
   pip install pytest pytest-cov pytest-mock
   
   # Create test directory structure
   mkdir -p tests/unit tests/integration
   ```

2. **Add core unit tests**
   ```python
   # Example test for config.py
   def test_config_initialization():
       config = Config('tests/fixtures/test_config.yaml')
       assert config.get('system.debug') is False
       assert config.get('workspace.default') == 'default'
   ```

3. **Create test fixtures**
   ```bash
   # Create example documents, vector stores, etc.
   mkdir -p tests/fixtures/data tests/fixtures/body
   ```

4. **Implement CI workflow**
   ```yaml
   # .github/workflows/test.yml
   name: Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - name: Set up Python
           uses: actions/setup-python@v4
           with:
             python-version: '3.9'
         - name: Install dependencies
           run: pip install -r requirements-dev.txt
         - name: Run tests
           run: pytest --cov=core
   ```

## 4. Error Handling and Logging

### Issues
- Inconsistent error logging
- Debug mode hard to control
- Missing structured logging

### Implementation Plan
1. **Implement structured logging**
   ```python
   # Enhance logging.py
   import logging
   import json
   
   class JsonFormatter(logging.Formatter):
       """JSON formatter for structured logging"""
       def format(self, record):
           log_record = {
               "timestamp": self.formatTime(record),
               "level": record.levelname,
               "message": record.getMessage(),
               "module": record.module,
               "function": record.funcName,
           }
           
           if hasattr(record, 'context'):
               log_record['context'] = record.context
               
           if record.exc_info:
               log_record['exception'] = self.formatException(record.exc_info)
               
           return json.dumps(log_record)
   ```

2. **Add context to logs**
   ```python
   # Add to relevant functions
   def process_command(self, command_string):
       """Process a command string"""
       logger = get_logger(__name__)
       logger.info("Processing command", extra={"context": {"command": command_string}})
   ```

3. **Create logging configuration file**
   ```yaml
   # logging_config.yaml
   version: 1
   formatters:
     json:
       (): core.engine.logging.JsonFormatter
     console:
       format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
   handlers:
     console:
       class: logging.StreamHandler
       formatter: console
     file:
       class: logging.FileHandler
       formatter: json
       filename: logs/monkey.log
   loggers:
     core:
       level: INFO
       handlers: [console, file]
       propagate: no
   root:
     level: INFO
     handlers: [console]
   ```

## 5. Deployment and Distribution

### Issues
- No packaging for end users
- Manual installation process
- Not easily deployable

### Implementation Plan
1. **Create Docker container**
   ```dockerfile
   # Dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   # Install system dependencies
   RUN apt-get update && apt-get install -y \
       build-essential \
       && rm -rf /var/lib/apt/lists/*
   
   # Copy requirements first for caching
   COPY requirements.txt .
   RUN pip install --no-cache-dir -r requirements.txt
   
   # Copy application code
   COPY . .
   
   # Create directories
   RUN mkdir -p data body logs lexicon
   
   # Set entrypoint
   ENTRYPOINT ["python", "monkey.py"]
   ```

2. **Create installation script**
   ```bash
   #!/bin/bash
   # install.sh
   
   set -e
   
   # Check Python version
   python_version=$(python3 --version | cut -d' ' -f2)
   required_version="3.8.0"
   
   if [[ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]]; then
       echo "Error: Python $required_version or higher is required (found $python_version)"
       exit 1
   fi
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -U pip
   pip install -r requirements.txt
   
   # Install the application
   pip install -e .
   
   echo "Installation complete. Run 'monkey' to start the application."
   ```

3. **Create binary distribution with PyInstaller**
   ```bash
   # Install PyInstaller
   pip install pyinstaller
   
   # Create spec file
   pyi-makespec --onefile --name monkey monkey.py
   
   # Edit monkey.spec to include data files
   # Add: datas=[('config.yaml', '.'), ('lexicon', 'lexicon')]
   
   # Build binary
   pyinstaller monkey.spec
   ```

## 6. Performance Optimization

### Issues
- Memory usage for large document sets
- No caching strategy
- Processing bottlenecks

### Implementation Plan
1. **Implement memory-efficient processing**
   ```python
   # Add to document processing
   def process_large_document(self, filepath, chunk_size=100000):
       """Process large documents in chunks"""
       with open(filepath, 'r', encoding='utf-8') as file:
           chunk = file.read(chunk_size)
           while chunk:
               # Process chunk
               self._process_chunk(chunk)
               # Get next chunk
               chunk = file.read(chunk_size)
   ```

2. **Add caching layer**
   ```python
   # Add caching decorator
   import functools
   import hashlib
   import pickle
   import os
   
   def cache_result(cache_dir='cache'):
       """Cache function results to disk"""
       def decorator(func):
           @functools.wraps(func)
           def wrapper(*args, **kwargs):
               # Create cache key from function name and arguments
               key_parts = [func.__name__]
               key_parts.extend([str(arg) for arg in args])
               key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
               key = hashlib.md5('|'.join(key_parts).encode()).hexdigest()
               
               cache_file = os.path.join(cache_dir, f"{key}.cache")
               
               # Try to load from cache
               if os.path.exists(cache_file):
                   try:
                       with open(cache_file, 'rb') as f:
                           return pickle.load(f)
                   except Exception:
                       pass
               
               # Calculate result
               result = func(*args, **kwargs)
               
               # Save to cache
               os.makedirs(cache_dir, exist_ok=True)
               with open(cache_file, 'wb') as f:
                   pickle.dump(result, f)
                   
               return result
           return wrapper
       return decorator
   ```

3. **Add parallel processing**
   ```python
   # Add to batch operations
   from concurrent.futures import ProcessPoolExecutor
   
   def process_documents_parallel(self, docs, max_workers=4):
       """Process documents in parallel"""
       results = []
       with ProcessPoolExecutor(max_workers=max_workers) as executor:
           futures = [executor.submit(self._process_single_document, doc) for doc in docs]
           for future in futures:
               results.append(future.result())
       return results
   ```

## 7. Security Enhancements

### Issues
- Limited input validation
- Path traversal vulnerabilities
- No authentication mechanism

### Implementation Plan
1. **Add input validation**
   ```python
   # Add validation module
   def validate_path(path):
       """Validate and sanitize a file path"""
       # Normalize path
       path = os.path.normpath(path)
       
       # Check for path traversal attempts
       if '..' in path or path.startswith('/'):
           raise ValueError("Invalid path: potential path traversal")
           
       # Check if path exists within authorized directories
       if not path.startswith(('data/', 'body/')):
           raise ValueError("Invalid path: outside of authorized directories")
           
       return path
   ```

2. **Add authentication for API use**
   ```python
   # If exposing as API, add basic auth
   def authenticate(username, password):
       """Authenticate a user"""
       # Load users from secure storage
       with open('config/users.yaml', 'r') as f:
           users = yaml.safe_load(f)
           
       if username not in users:
           return False
           
       # Use secure password verification
       import hashlib
       stored_hash = users[username]['password_hash']
       salt = users[username]['salt']
       
       # Calculate hash with provided password and stored salt
       calculated_hash = hashlib.pbkdf2_hmac(
           'sha256', 
           password.encode(), 
           salt.encode(), 
           100000
       ).hex()
       
       return calculated_hash == stored_hash
   ```

3. **Implement secure configuration**
   ```python
   # Enhance config.py to handle sensitive data
   def get_secure(self, path, default=None):
       """Get a secure configuration value, with environment fallback"""
       # Try environment variable first (upper case with underscores)
       env_var = path.replace('.', '_').upper()
       env_value = os.environ.get(env_var)
       if env_value:
           return env_value
           
       # Then try config file
       return self.get(path, default)
   ```

## 8. Monitoring and Observability

### Issues
- Limited performance monitoring
- No health checks
- Difficult to troubleshoot in production

### Implementation Plan
1. **Add performance metrics**
   ```python
   # Add metrics.py
   import time
   from collections import defaultdict
   
   class Metrics:
       """Simple metrics collection"""
       def __init__(self):
           self.counters = defaultdict(int)
           self.timers = defaultdict(list)
           
       def increment(self, metric, value=1):
           """Increment a counter"""
           self.counters[metric] += value
           
       def timing(self, metric):
           """Context manager for timing operations"""
           class TimingContext:
               def __init__(self, metrics, metric):
                   self.metrics = metrics
                   self.metric = metric
                   
               def __enter__(self):
                   self.start = time.time()
                   return self
                   
               def __exit__(self, *args):
                   duration = time.time() - self.start
                   self.metrics.timers[self.metric].append(duration)
                   
           return TimingContext(self, metric)
           
       def get_stats(self):
           """Get current metrics"""
           stats = {
               'counters': dict(self.counters)
           }
           
           # Calculate timer statistics
           timer_stats = {}
           for metric, values in self.timers.items():
               if not values:
                   continue
                   
               timer_stats[metric] = {
                   'count': len(values),
                   'avg': sum(values) / len(values),
                   'min': min(values),
                   'max': max(values)
               }
               
           stats['timers'] = timer_stats
           return stats
   ```

2. **Add health check endpoint**
   ```python
   # If exposing as service
   def health_check():
       """Check system health"""
       checks = {
           "storage": check_storage_access(),
           "llm": check_llm_connection(),
           "disk_space": check_disk_space()
       }
       
       all_healthy = all(checks.values())
       
       return {
           "status": "healthy" if all_healthy else "unhealthy",
           "checks": checks,
           "timestamp": datetime.datetime.now().isoformat()
       }
   ```

3. **Integrate with monitoring tools**
   ```python
   # Add Prometheus metrics if needed
   from prometheus_client import Counter, Histogram, start_http_server
   
   # Define metrics
   DOCUMENT_COUNTER = Counter(
       'monkey_documents_processed_total', 
       'Number of documents processed'
   )
   
   QUERY_LATENCY = Histogram(
       'monkey_query_latency_seconds',
       'Time to process queries',
       buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
   )
   
   # Expose metrics server
   def start_metrics_server(port=8000):
       """Start Prometheus metrics server"""
       start_http_server(port)
       print(f"Metrics server started on port {port}")
   ```

## 9. Binary Compilation Options

### Approach 1: PyInstaller (Best for Initial Deployment)
```bash
# Install PyInstaller
pip install pyinstaller

# Create a customized spec file
pyi-makespec --onefile --name monkey monkey.py

# Customize the spec file for better handling of ML dependencies
# Edit monkey.spec to:
# 1. Add data files
# 2. Handle hidden imports for ML libraries
# 3. Configure binary compression

# Build the binary
pyinstaller monkey.spec
```

### Approach 2: Cython for Performance-Critical Modules
```python
# setup.py with Cython extensions
from setuptools import setup, Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "core.engine.fast_processing",
        ["core/engine/fast_processing.pyx"],
        extra_compile_args=["-O3"]
    )
]

setup(
    name="monkey",
    ext_modules=cythonize(extensions),
    # Other setup parameters
)
```

### Approach 3: Docker-based Deployment (Most Reliable)
```bash
# Build Docker image
docker build -t monkey-toolkit .

# Run application
docker run -v ./data:/app/data -v ./body:/app/body -it monkey-toolkit

# Create distribution script
cat > distribute.sh << 'EOF'
#!/bin/bash
# Pack everything needed to run the application
mkdir -p dist
tar -czf dist/monkey-toolkit.tar.gz Dockerfile requirements.txt core/ config.yaml install.sh
echo "Created distribution package at dist/monkey-toolkit.tar.gz"
EOF

chmod +x distribute.sh
```

## 10. Implementation Timeline

### Phase 1: Code Quality (Week 1-2)
- Apply code formatting
- Standardize error handling
- Add complete type hints
- Create dependency management

### Phase 2: Testing (Week 3-4)
- Set up testing framework
- Add core unit tests
- Create test fixtures
- Implement CI workflow

### Phase 3: Performance and Security (Week 5-6)
- Add caching layer
- Implement parallel processing
- Add input validation
- Secure configuration management

### Phase 4: Deployment and Distribution (Week 7-8)
- Create Docker container
- Build binary with PyInstaller
- Create installation script
- Add monitoring and metrics

### Phase 5: Documentation and Training (Week 9-10)
- Update documentation
- Create user guides
- Develop quick start tutorial
- Train team on deployment processes