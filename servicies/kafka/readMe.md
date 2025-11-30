### Apache Kafka

## What is Kafka?

Apache Kafka is a distributed streaming platform that enables publishing, storing, and processing data streams in real-time. It was originally developed by LinkedIn and is now an open-source Apache project.

**Key characteristics:**
- Distributed message broker
- Persistent storage for messages
- Pub/Sub model (publish-subscribe)
- High throughput (millions of messages/second)

## Why is Kafka good?

1. **Scalability** - Scales horizontally by adding brokers
2. **Durability** - Messages are persisted to disk and replicated
3. **Performance** - Very low latency (<10ms)
4. **Fault tolerance** - Continues to work even if some nodes fail
5. **Replay** - You can re-read messages from the past
6. **Decoupling** - Producers and consumers are independent

## How does it work internally?

### Core concepts:

```
Producer → [Topic] → Consumer
              ↓
         Partitions (P0, P1, P2...)
              ↓
         Replicas (Leader + Followers)
```

**Topics** - Categories/channels for messages
**Partitions** - Subdivisions of a topic for parallelism
**Brokers** - Kafka servers that store data
**Consumer Groups** - Groups of consumers that process in parallel
**Offsets** - Current position of a consumer in a partition

### Data flow:

1. Producer sends message to a Topic
2. Message is written to Leader partition
3. Followers replicate the data
4. Consumer reads messages in order
5. Offset is updated

## How to use Kafka

### Installation with Docker:

```yaml
# docker-compose.yaml
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

### Python example with confluent-kafka:

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

## ZooKeeper vs KRaft Mode

### ZooKeeper Mode (Legacy)

ZooKeeper is a distributed coordination service previously used by Kafka for:
- Cluster management
- Leader election for partitions
- Metadata storage

**ZooKeeper Architecture:**
- Stores data in a hierarchical structure (like a file system)
- Each data register is called a **znode**

**Types of znodes:**
- **Persistent** - Remains until explicitly deleted
- **Ephemeral** - Automatically deleted when session disconnects. Cannot have children
- **Sequential** - Creates sequential IDs automatically

**Characteristics:**
- Sequential consistency for all updates
- Does not allow concurrent writes
- Client always sees the same view of the service

### KRaft Mode (New - Recommended)

KRaft (Kafka Raft) eliminates the ZooKeeper dependency by:
- Metadata is stored directly in Kafka
- Uses Raft protocol for consensus
- Simpler to configure and operate

**KRaft advantages:**
- Simplified architecture (single system)
- Faster startup
- Improved scalability
- Easier operations

```yaml
# KRaft mode config (no ZooKeeper)
environment:
  - KAFKA_CFG_PROCESS_ROLES=controller,broker
  - KAFKA_CFG_CONTROLLER_QUORUM_VOTERS=0@kafka:9093
```

**When to use what:**
- **KRaft** - For new installations (Kafka 3.3+)
- **ZooKeeper** - For legacy systems that haven't migrated yet

---

## Bank App Example

See the `bank_app/` folder for a complete banking application example with:
- 2 INPUT topics (transaction_requests, account_updates)
- 5 OUTPUT topics (validated, fraud, balance, profile, notifications)
- Pydantic models for Request/Response
- Configurable Consumer and Producer