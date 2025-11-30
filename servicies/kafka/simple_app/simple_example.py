from typing import Optional
from confluent_kafka import Consumer, Producer
from pydantic import BaseSettings, Field

#################


class KafkaConfig(BaseSettings):
    bootstrap_servers: str = Field(..., env="KAFKA_BOOTSTRAP_SERVERS")
    client_id: Optional[str] = Field(None, env="KAFKA_CLIENT_ID")

    def get_dict(self):
        return {
            name.replace("_", "."): value for name, value in self.model_dump().items()
        }


class KafkaConsumerConfig(KafkaConfig):
    group_id: str = Field(..., env="KAFKA_CONSUMER_GROUP")
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = True
    session_timeout_ms: int = 30000
    max_poll_interval_ms: int = 1800000


class KafkaProducerConfig(KafkaConfig):
    compression_type: str = "gzip"
    linger_ms: int = 10
    batch_size: int = 32768


class KafkaTopics(BaseSettings):
    consumer_topic: str = Field(..., env="KAFKA_CONSUMER_TOPIC")
    producer_topic: str = Field(..., env="KAFKA_PRODUCER_TOPIC")


###############


class KafkaManager:
    def __init__(
        self,
        consumer_config: KafkaConsumerConfig,
        producer_config: KafkaProducerConfig,
        topics: KafkaTopics,
    ):
        self.topics = topics

        self.consumer = Consumer(consumer_config.get_dict())
        self.consumer.subscribe([self.topics.consumer_topic])

        self.producer = Producer(producer_config.get_dict())

    def poll(self, timeout: float = 1.0) -> Optional[str]:
        self.producer.poll(0)

        msg = self.consumer.poll(timeout=timeout)
        if msg is None:
            return None

        if msg.error():
            return None

        return msg.value()

    def produce(self, message: str, topic: Optional[str] = None) -> None:
        target_topic = topic or self.topics.producer_topic
        self.producer.produce(target_topic, message.encode("utf-8"))

    def flush(self):
        self.producer.flush()

    def close(self):
        self.flush()
        self.consumer.close()
