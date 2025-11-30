# Redis

## What is Redis?

Redis is a popular in-memory cache system for storing data in efficient data structures with O(1) retrieval time.

## When to use Redis?

Consider Redis when you encounter these scenarios:

- **Reused values** - Your application repeatedly accesses the same data
- **System instability** - The system fails unpredictably due to external issues
- **State persistence** - You need to maintain user data across crashes or disconnections (e.g., user rejoins within 10-20 seconds)

Redis solves all three problems.

## Example Use Case

See `simple.py` for a practical example: a service that waits for 3 components (audio, video, image) to be completed. Each component is stored as a class instance indexed by a request ID. The system checks if all components are present before returning the complete response.

## Basic Concepts

**In-memory storage** - Data lives in RAM for fast access (nanoseconds vs milliseconds on disk)

**Key-value pairs** - Access data by key: `GET user:123` → returns user data

**Data types** - Strings, Lists, Hashes, Sets, Sorted Sets (each optimized for different use cases)

**TTL (Time To Live)** - Automatically delete keys after a specified time

**Persistence** - Optional: save to disk (AOF or RDB snapshots)

## Common Use Cases

- Session storage (keep user logged-in state)
- Cache layer (store database query results)
- Rate limiting (count requests per user)
- Real-time analytics (counters, leaderboards)
- Task queues (job processing)

