from functools import lru_cache
import time


@lru_cache(maxsize=128)
def conversion_usd_euro(usd):
    time.sleep(1)
    return usd * 0.92


@lru_cache(maxsize=32)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)


@lru_cache(maxsize=256)
def expensive_calculation(x, y):
    time.sleep(2)
    return x ** y + x * y


if __name__ == "__main__":
    print("First call (slow, 1 second):")
    start = time.time()
    result1 = conversion_usd_euro(100)
    print(f"100 USD = {result1} EUR | Time: {time.time() - start:.2f}s")
    
    print("\nSecond call (fast, cached):")
    start = time.time()
    result2 = conversion_usd_euro(100)
    print(f"100 USD = {result2} EUR | Time: {time.time() - start:.4f}s")
    
    print("First call (slow):")
    start = time.time()
    fib1 = fibonacci(300)
    print(f"fibonacci(300) = {fib1} | Time: {time.time() - start:.2f}s")
    
    print("\nSecond call (fast, cached):")
    start = time.time()
    fib2 = fibonacci(300)
    print(f"fibonacci(300) = {fib2} | Time: {time.time() - start:.4f}s")
    
    print("\nExtra INFO")
    print(f"Fibonacci cache: {fibonacci.cache_info()}")
    
    print("\nExtra cache conversion info")
    print(f"Conversion cache: {conversion_usd_euro.cache_info()}")
    print(f"conversion_usd_euro(50) = {conversion_usd_euro(50)}")
    print(f"conversion_usd_euro(100) = {conversion_usd_euro(300)}")
    print(f"Conversion cache: {conversion_usd_euro.cache_info()}")

