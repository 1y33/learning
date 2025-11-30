from typing import Optional
from confluent_kafka import Consumer , Producer

from pydantic import BaseSettings, Field,BaseModel

class KafkaConfig(BaseSettings):
    bootstrap_servers: str = Field(..., env="KAFKA_BOOTSTRAP_SERVERS")
    client_id: Optional[str] = Field(None, env="KAFKA_CLIENT_ID")

    def get_dict(self):
        return {name.replace("_", "."): value for name, value in self.model_dump().items()}

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