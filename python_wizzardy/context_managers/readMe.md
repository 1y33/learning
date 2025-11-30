# Context Managers

## What is it?

Context managers handle setup and teardown of resources. You've seen them with `with` statements:

```python
with open('file.txt') as f:
    data = f.read()
# File automatically closed here
```

The file is opened (`__enter__`), used, then automatically closed (`__exit__`). No need to manually close it.

## Two ways to create them

1. **Class-based** - Implement `__enter__` and `__exit__` methods
2. **Decorator-based** - Use `@contextmanager` from `contextlib`

---

## Method 1: Class-based

```python
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection = None
        self.connection_string = connection_string
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}...")
        self.connection = f"Connected to {self.connection_string}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection...")
        self.connection = None
```

**Real-world example 1: Database connection**
```python
with DatabaseConnection("postgres://localhost/mydb") as conn:
    print(f"Using: {conn}")
    # Do database operations
# Connection automatically closed
```

**Real-world example 2: File handling**
```python
class FileHandler:
    def __init__(self, filename, mode='r'):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        self.file = open(self.filename, self.mode)
        print(f"Opened {self.filename}")
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
            print(f"Closed {self.filename}")

with FileHandler("data.txt", "r") as f:
    content = f.read()
# File automatically closed
```

---

## Method 2: Decorator-based (@contextmanager)

```python
from contextlib import contextmanager

@contextmanager
def database_connection(connection_string):
    print(f"Connecting to {connection_string}...")
    connection = f"Connected to {connection_string}"
    try:
        yield connection
    finally:
        print("Closing connection...")
```

**Real-world example 1: API request with timeout**
```python
from contextlib import contextmanager
import requests
import time

@contextmanager
def api_request_timer():
    start = time.time()
    print("Starting API request...")
    try:
        yield
    finally:
        elapsed = time.time() - start
        print(f"Request completed in {elapsed:.2f}s")

with api_request_timer():
    # Simulate API call
    time.sleep(1)
    print("Processing data...")
# Automatically prints elapsed time
```

**Real-world example 2: Temporary directory cleanup**
```python
from contextlib import contextmanager
import os
import tempfile

@contextmanager
def temporary_directory():
    tmpdir = tempfile.mkdtemp()
    print(f"Created temp directory: {tmpdir}")
    try:
        yield tmpdir
    finally:
        import shutil
        shutil.rmtree(tmpdir)
        print(f"Cleaned up temp directory")

with temporary_directory() as tmpdir:
    # Create files in tmpdir
    filepath = os.path.join(tmpdir, "temp_file.txt")
    with open(filepath, 'w') as f:
        f.write("temp data")
    print(f"File created in {tmpdir}")
# Directory and all files automatically deleted
```

---

## When to use context managers

- Opening/closing files or connections
- Acquiring/releasing locks
- Starting/stopping timers
- Creating/cleaning up temporary resources
- Database transactions (begin/commit/rollback)