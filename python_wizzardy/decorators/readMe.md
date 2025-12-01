# Python Decorators

## What is a Decorator?

A decorator wraps a function to add extra behavior. The `@something` above a function is a decorator.

```python
@timer
def my_function():
    pass

# Same as:
my_function = timer(my_function)
```

---

## Simple Decorator (No Arguments)

2 levels of nesting:

```python
def timer(func):              # Takes a function
    def wrapper(*args, **kwargs):
        # extra behavior
        result = func(*args, **kwargs)
        return result
    return wrapper            # Returns wrapper
```

---

## Decorator WITH Arguments

When you want `@retry(max_attempts=3)`, you need **3 levels**:

```python
def retry(max_attempts=3):       # Level 1: Takes CONFIG
    def decorator(func):          # Level 2: Takes FUNCTION
        def wrapper(*args):       # Level 3: Takes ARGUMENTS
            # logic here
            return func(*args)
        return wrapper
    return decorator
```

### Why 3 levels?

```
@retry(max_attempts=3)   → Step 1: retry(3) returns decorator
def flaky():             → Step 2: decorator(flaky) returns wrapper
    pass                 → Step 3: wrapper() runs when you call flaky()
```

### Visual

```
retry(config) → decorator(func) → wrapper(args)
     ↓              ↓                 ↓
  settings      your function     actual call
```

---

## Key Rules

1. **`@wraps(func)`** - Preserves original function's name/docstring
2. **`*args, **kwargs`** - Pass any arguments to wrapped function
3. **Returns go outward** - wrapper returns result, decorator returns wrapper, outer returns decorator
