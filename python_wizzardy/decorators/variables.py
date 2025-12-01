import time
from functools import wraps
import random


def retry(max_attempts: int = 3, delay: float = 1.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    if attempt < max_attempts:
                        time.sleep(delay)
            raise last_exception
        return wrapper      # returns wrapper to decorator
    return decorator        # returns decorator to retry


# Test it
@retry(max_attempts=3, delay=0.5)
def flaky_api_call():
    if random.random() < 0.7:  # 70% failure rate
        raise ConnectionError("Server unavailable")
    return {"status": "success"}


if __name__ == "__main__":
    try:
        result = flaky_api_call()
        print(f"Success: {result}")
    except ConnectionError as e:
        print(f"All attempts failed: {e}")
