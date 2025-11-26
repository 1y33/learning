# Docker Services Documentation

## Overview

This project uses Docker to manage multiple services. The key components are:

- **Services**: Docker containers and applications you want to run
- **Volumes**: Persistent data storage for containers

## Services Structure

A typical service definition includes image, container name, ports, volumes, and health checks.

### Basic Service Example

```yaml
services:
  redis:
    image: redis:7-alpine
    container_name: redis_container
    command: redis-server --appendonly yes
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
```

### What This Means

| Component | Purpose |
|-----------|---------|
| `image` | Redis 7 lightweight version |
| `container_name` | Identifies the running container |
| `command` | Enables AOF (Append Only File) persistence |
| `ports` | Maps container port 6379 to host port 6379 |
| `volumes` | Stores data persistently in `redis-data` volume at `/data` |
| `healthcheck` | Monitors container health using Redis ping command |

#### Health Check Deep Dive

The `healthcheck` block tells Docker how to verify if the service is running correctly:

- **test**: The command to run. `["CMD", "redis-cli", "ping"]` tries to ping Redis
- **interval**: How often to run the health check (every 10 seconds)
- **timeout**: Maximum time to wait for the check to complete (5 seconds). If the command takes longer, it fails
- **retries**: Number of consecutive failures before marking container as unhealthy (5 failures = unhealthy)

**Example Timeline:**
- First failure: container is still "starting"
- Second failure: still "starting"
- ...continues...
- After 5 consecutive failures: container marked as "unhealthy"
- Docker can restart it or other services can refuse to start until it recovers

## Service Dependencies

Services can depend on other services. Use `depends_on` to ensure proper startup order and health status.

### Example: Kafka UI Depending on Kafka

```yaml
services:
  kafka:
    image: confluentinc/cp-kafka:latest
    container_name: kafka
    healthcheck:
      test: ["CMD", "kafka-broker-api-versions", "--bootstrap-server", "localhost:9092"]
      interval: 10s
      timeout: 5s
      retries: 5

  kafka-ui:
    image: provectuslabs/kafka-ui:latest
    container_name: kafka_ui
    depends_on:
      kafka:
        condition: service_healthy
    ports:
      - "8080:8080"
```

The `kafka-ui` service will only start after `kafka` passes its health check.
