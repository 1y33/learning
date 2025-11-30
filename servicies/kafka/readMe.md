# Apache Kafka

## What is it?

Kafka is basically a message queue on steroids. Think of it like a post office that never loses mail and can handle millions of letters per second. 

LinkedIn built it because they needed something fast to move data between their services. Now everyone uses it - Netflix, Uber, banks, you name it.

The idea is simple: one service sends a message, another service reads it. But unlike a normal queue, Kafka keeps the messages around so you can read them again if needed.


## Why use it?

Its fast. Really fast. We're talking millions of messages per second.

It doesnt lose data. Messages are saved to disk and copied to multiple servers. If one server dies, the others still have your data.

Services dont need to know about each other. The producer just throws messages into a topic, consumers read when they want. They dont care about each other.

You can replay old messages. Made a bug? Fix it and reprocess the last week of data. Normal queues delete messages after reading, Kafka keeps them.


## How it works

You have Topics - these are like folders or channels. "orders", "payments", "user-events", whatever you want.

Producers send messages to topics. Consumers read from topics. Thats it.

Under the hood, each topic is split into Partitions for speed. Messages go to a Leader partition first, then get copied to Followers. If the leader dies, a follower takes over.

Consumer Groups let multiple consumers split the work. If you have 3 partitions and 3 consumers in a group, each consumer handles one partition.

Offsets track where each consumer is in the topic. Like a bookmark.


## Running it

Easiest way is Docker. Heres a simple setup:

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

Python code is straightforward:

```python
from confluent_kafka import Producer, Consumer

# Send a message
producer = Producer({'bootstrap.servers': 'localhost:9092'})
producer.produce('my-topic', b'Hello!')
producer.flush()

# Read messages
consumer = Consumer({
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'my-group',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['my-topic'])

while True:
    msg = consumer.poll(1.0)
    if msg:
        print(msg.value())
```


## ZooKeeper vs KRaft

Old Kafka needed ZooKeeper - a separate service that kept track of which broker is the leader, what topics exist, etc. Pain to manage two systems.

ZooKeeper stores stuff in a tree structure with nodes called znodes. Some are permanent, some disappear when the client disconnects. It makes sure everyone agrees on who the leader is.

New Kafka (3.3+) uses KRaft mode. No more ZooKeeper. Kafka handles its own metadata now using the Raft consensus protocol. One less thing to babysit.

If youre starting fresh, use KRaft. If you have old Kafka with ZooKeeper, it still works but consider migrating eventually.


## Bank App Example

Check the `bank_app/` folder. Its a realistic example with 2 input topics and 5 output topics, showing how a banking system might process transactions through Kafka.