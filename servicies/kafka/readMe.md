# Apache Kafka

## What is Kafka?

Apache Kafka is a distributed streaming platform for publishing, storing, and processing data streams in real-time. Originally developed by LinkedIn, it's now an open-source Apache project used by Netflix, Uber, and most major tech companies.

The core idea: producers send messages to topics, consumers read from topics. Unlike traditional queues, Kafka persists messages so they can be replayed.

## Why Kafka?

**Performance** - Handles millions of messages per second with sub-10ms latency.

**Durability** - Messages are persisted to disk and replicated across multiple brokers. If one server fails, others still have your data.

**Decoupling** - Producers and consumers don't need to know about each other. They just interact with topics.

**Replay** - Messages aren't deleted after consumption. You can reprocess historical data anytime.

**Scalability** - Add more brokers to handle increased load horizontally.

## Core Concepts

**Topics** - Named channels where messages are published. Think of them as categories.

**Partitions** - Topics are split into partitions for parallelism. Each partition is an ordered, immutable log.

**Brokers** - Kafka servers that store partitions. A cluster typically has multiple brokers.

**Producers** - Applications that publish messages to topics.

**Consumers** - Applications that read messages from topics.

**Consumer Groups** - Multiple consumers that share the workload. Each partition is consumed by only one consumer in the group.

**Offsets** - Position markers tracking where each consumer is in a partition.

## How Data Flows

1. Producer sends a message to a topic
2. Message is written to the leader partition
3. Follower replicas copy the data
4. Consumer reads messages and commits offset
5. If leader fails, a follower becomes the new leader

## Running Kafka with Docker

```yaml
services:
  kafka:
    image: bitnami/kafka:latest
    ports:
      - "9092:9092"
    environment:
      - KAFKA_CFG_NODE_ID=0
      - KAFKA_CFG_PROCESS_ROLES=controller,broker
      - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka:9093
      - KAFKA_CFG_LISTENERS=PLAINTEXT://:9092,CONTROLLER://:9093
      - KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092
      - KAFKA_CFG_CONTROLLER_LISTENER_NAMES=CONTROLLER
```

## Python Example

```python
from confluent_kafka import Producer, Consumer

# Producer
producer = Producer({'bootstrap.servers': 'localhost:9092'})
producer.produce('my-topic', b'Hello Kafka!')
producer.flush()

# Consumer
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['my-topic'])

while True:
    msg = consumer.poll(1.0)
    if msg:
        print(msg.value().decode('utf-8'))
```

---

## ZooKeeper vs KRaft

### ZooKeeper Mode (Legacy)

ZooKeeper is a distributed coordination service that Kafka traditionally used for cluster management, leader election, and metadata storage.

ZooKeeper stores data in a hierarchical tree structure. Each node is called a **znode**:
- **Persistent** - Remains until explicitly deleted
- **Ephemeral** - Deleted when the client session disconnects
- **Sequential** - Automatically assigns sequential IDs

ZooKeeper ensures sequential consistency and prevents concurrent writes. All clients see the same view of the cluster state.

### KRaft Mode (Recommended)

KRaft (Kafka Raft) eliminates the ZooKeeper dependency. Introduced in Kafka 2.8 and production-ready since 3.3.

Metadata is now stored directly in Kafka using the Raft consensus protocol. This simplifies architecture, speeds up startup, and reduces operational complexity.

```yaml
# KRaft configuration
environment:
  - KAFKA_CFG_PROCESS_ROLES=controller,broker
  - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka:9093
```

**Use KRaft** for new deployments. **Use ZooKeeper** only for legacy systems not yet migrated.

---

## Bank App Example

See `bank_app/` for a complete example:
- 2 input topics: `transaction_requests`, `account_updates`
- 5 output topics: `transactions_validated`, `fraud_alerts`, `account_balance_updates`, `account_profile_updates`, `notifications`
- Pydantic models for request/response validation
- Configurable consumer and producer classes