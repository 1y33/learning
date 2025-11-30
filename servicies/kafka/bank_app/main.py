from typing import Type, Callable, List
from confluent_kafka import Consumer, Producer
from pydantic import BaseModel
from decimal import Decimal
import json
import uuid

from kafka_configs import KafkaConsumerConfig, KafkaProducerConfig
from api import (
    MessageType,
    TransactionRequest, TransactionType,
    AccountUpdateRequest, AccountUpdateType,
    RequestMessage, ResponseMessage,
    TransactionValidatedResponse,
    FraudAlertResponse,
    BalanceUpdateResponse,
    ProfileUpdateResponse,
    NotificationResponse,
    Currency,
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

    def poll(self, timeout: float = 1.0) -> RequestMessage | None:
        msg = self.consumer.poll(timeout=timeout)
        if msg is None or msg.error():
            return None

        topic = msg.topic()
        data = json.loads(msg.value().decode("utf-8"))

        request_msg = RequestMessage.model_validate(data)

        if topic in self._handlers:
            self._handlers[topic](request_msg)

        return request_msg

    def close(self):
        self.consumer.close()


class BankProducer:
    def __init__(self, config: KafkaProducerConfig):
        self.config = config
        self.producer = Producer(self.config.get_dict())

    def produce(self, topic: str, message: ResponseMessage):
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


if __name__ == "__main__":
    consumer_config = KafkaConsumerConfig(
        bootstrap_servers="localhost:9092", group_id="bank-processor"
    )
    producer_config = KafkaProducerConfig(bootstrap_servers="localhost:9092")

    consumer = BankConsumer(consumer_config, INPUT_TOPICS)
    producer = BankProducer(producer_config)

    def handle_transaction(msg: RequestMessage):
        tx = msg.request
        print(f"[TX] Type: {msg.message_type} | ID: {tx.transaction_id}")

        validated = ResponseMessage(
            message_type=MessageType.TRANSACTION_VALIDATED,
            response=TransactionValidatedResponse(
                transaction_id=tx.transaction_id,
                status="approved",
                reference_number=f"REF{uuid.uuid4().hex[:8].upper()}"
            )
        )
        producer.produce(OUTPUT_TOPICS["validated"], validated)

        fraud_check = ResponseMessage(
            message_type=MessageType.FRAUD_ALERT,
            response=FraudAlertResponse(
                transaction_id=tx.transaction_id,
                risk_score=0.15,
                risk_level="low",
                reasons=["Normal transaction pattern"],
                action="allow"
            )
        )
        producer.produce(OUTPUT_TOPICS["fraud"], fraud_check)

        balance = ResponseMessage(
            message_type=MessageType.BALANCE_UPDATE,
            response=BalanceUpdateResponse(
                account_id=tx.metadata.account_id,
                old_balance=Decimal("10000.00"),
                new_balance=Decimal("10000.00") - tx.amount,
                currency=tx.currency,
                transaction_id=tx.transaction_id
            )
        )
        producer.produce(OUTPUT_TOPICS["balance"], balance)

        notify = ResponseMessage(
            message_type=MessageType.NOTIFICATION,
            response=NotificationResponse(
                user_id=tx.metadata.account_id,
                channel="sms",
                message=f"Transfer de {tx.amount} {tx.currency} efectuat cu succes."
            )
        )
        producer.produce(OUTPUT_TOPICS["notify"], notify)

    def handle_account_update(msg: RequestMessage):
        update = msg.request
        print(f"[ACCOUNT] Type: {msg.message_type} | ID: {update.account_id}")

        profile = ResponseMessage(
            message_type=MessageType.PROFILE_UPDATE,
            response=ProfileUpdateResponse(
                account_id=update.account_id,
                update_type=update.update_type,
                status="success",
                details="Profile updated successfully"
            )
        )
        producer.produce(OUTPUT_TOPICS["profile"], profile)

        notify = ResponseMessage(
            message_type=MessageType.NOTIFICATION,
            response=NotificationResponse(
                user_id=update.user_id,
                channel="email",
                message=f"Contul tau a fost actualizat: {update.update_type}"
            )
        )
        producer.produce(OUTPUT_TOPICS["notify"], notify)

    consumer.register_handler("transaction_requests", handle_transaction)
    consumer.register_handler("account_updates", handle_account_update)

    print("Bank processor started. Waiting for messages...")
    try:
        while True:
            consumer.poll(timeout=1.0)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        consumer.close()
        producer.close()
