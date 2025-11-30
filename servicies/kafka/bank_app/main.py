from typing import Type, Callable, List
from confluent_kafka import Consumer, Producer
from pydantic import BaseModel
import json

from kafka_configs import KafkaConsumerConfig, KafkaProducerConfig
from api import (
    TransactionRequest, TransferRequest, WithdrawalRequest, DepositRequest,
    AccountUpdateRequest, ChangeNameRequest, CreateCardRequest, CloseAccountRequest
)


class BankConsumer:
    def __init__(self, config: KafkaConsumerConfig, topics: List[str]):
        self.config = config
        self.topics = topics
        self.consumer = Consumer(self.config.get_dict())
        self.consumer.subscribe(self.topics)
        self._handlers: dict[str, Callable] = {}

    def register_handler(self, topic: str, handler: Callable[[BaseModel], None]):
        self._handlers[topic] = handler

    def poll(self, timeout: float = 1.0) -> BaseModel | None:
        msg = self.consumer.poll(timeout=timeout)
        if msg is None or msg.error():
            return None
        
        topic = msg.topic()
        data = json.loads(msg.value().decode("utf-8"))
        
        model = self._parse_message(topic, data)
        
        if topic in self._handlers:
            self._handlers[topic](model)
        
        return model

    def _parse_message(self, topic: str, data: dict) -> BaseModel:
        topic_models = {
            "transaction_requests": self._parse_transaction,
            "account_updates": self._parse_account_update,
        }
        parser = topic_models.get(topic)
        if parser:
            return parser(data)
        return data

    def _parse_transaction(self, data: dict) -> TransactionRequest:
        type_map = {
            "transfer": TransferRequest,
            "withdrawal": WithdrawalRequest,
            "deposit": DepositRequest,
        }
        tx_type = data.get("transaction_type", "transfer")
        model_class = type_map.get(tx_type, TransactionRequest)
        return model_class.model_validate(data)

    def _parse_account_update(self, data: dict) -> AccountUpdateRequest:
        type_map = {
            "change_name": ChangeNameRequest,
            "create_card": CreateCardRequest,
            "close_account": CloseAccountRequest,
        }
        update_type = data.get("update_type", "change_name")
        model_class = type_map.get(update_type, AccountUpdateRequest)
        return model_class.model_validate(data)

    def close(self):
        self.consumer.close()


class BankProducer:
    def __init__(self, config: KafkaProducerConfig):
        self.config = config
        self.producer = Producer(self.config.get_dict())

    def produce(self, topic: str, message: BaseModel):
        payload = message.model_dump_json()
        self.producer.produce(topic, payload.encode("utf-8"))
        self.producer.poll(0)

    def flush(self, timeout: float = 10.0):
        self.producer.flush(timeout)

    def close(self):
        self.flush()


INPUT_TOPICS = ["transaction_requests", "account_updates"]

OUTPUT_TOPICS = {
    "validated": "transactions_validated",
    "fraud": "fraud_alerts",
    "balance": "account_balance_updates",
    "profile": "account_profile_updates",
    "notify": "notifications",
}
        
        
