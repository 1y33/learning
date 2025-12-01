import time
from functools import wraps


def timer(func):
    @wraps(func)
    def wrapper(*args,**kwargs):
        start = time.time()
        result = func(*args,**kwargs)
        elapsed = time.time() - start
        print(f"{func.__name__} took { elapsed }")
        return result
    return wrapper


@timer
def slow_function():
    time.sleep(1)
    
if __name__ =="__main__":
    slow_function();