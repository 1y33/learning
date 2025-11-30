from decimal import Decimal
from datetime import datetime
from api import (
    MessageType,
    RequestMetadata,
    TransactionRequest,
    TransactionType,
    TransferData,
    WithdrawalData,
    DepositData,
    AccountUpdateRequest,
    AccountUpdateType,
    ChangeNameData,
    CreateCardData,
    CloseAccountData,
    Currency,
    CardType,
    RequestMessage,
    ResponseMessage,
    TransactionValidatedResponse,
    FraudAlertResponse,
    BalanceUpdateResponse,
    ProfileUpdateResponse,
    NotificationResponse,
)

metadata = RequestMetadata(
    account_id="RO12ABCD1234567890123456",
    user_ip="192.168.1.100",
    device_id="mobile_app_v2",
    user_agent="BankApp/2.0 Android",
    location="Bucharest, RO",
)

transfer_req = RequestMessage(
    message_type=MessageType.TRANSACTION_REQUEST,
    request=TransactionRequest(
        transaction_id="TXN001",
        transaction_type=TransactionType.TRANSFER,
        amount=Decimal("5000.00"),
        currency=Currency.RON,
        metadata=metadata,
        transfer=TransferData(
            to_account="RO89DEFG9876543210123456", beneficiary_name="Maria Ionescu"
        ),
    ),
)

withdrawal_req = RequestMessage(
    message_type=MessageType.TRANSACTION_REQUEST,
    request=TransactionRequest(
        transaction_id="TXN002",
        transaction_type=TransactionType.WITHDRAWAL,
        amount=Decimal("2000.00"),
        currency=Currency.RON,
        metadata=metadata,
        withdrawal=WithdrawalData(atm_id="ATM_CENTER_001"),
    ),
)

deposit_req = RequestMessage(
    message_type=MessageType.TRANSACTION_REQUEST,
    request=TransactionRequest(
        transaction_id="TXN003",
        transaction_type=TransactionType.DEPOSIT,
        amount=Decimal("10000.00"),
        currency=Currency.EUR,
        metadata=metadata,
        deposit=DepositData(source="cash_counter"),
    ),
)

change_name_req = RequestMessage(
    message_type=MessageType.ACCOUNT_UPDATE_REQUEST,
    request=AccountUpdateRequest(
        account_id="RO12ABCD1234567890123456",
        user_id="USER_456",
        update_type=AccountUpdateType.CHANGE_NAME,
        change_name=ChangeNameData(old_name="Ion Popescu", new_name="Ion P. Popescu"),
    ),
)

validated_resp = ResponseMessage(
    message_type=MessageType.TRANSACTION_VALIDATED,
    response=TransactionValidatedResponse(
        transaction_id="TXN001", status="approved", reference_number="REF2025113000001"
    ),
)

fraud_resp = ResponseMessage(
    message_type=MessageType.FRAUD_ALERT,
    response=FraudAlertResponse(
        transaction_id="TXN001",
        risk_score=0.85,
        risk_level="high",
        reasons=["Amount exceeds daily average", "New beneficiary"],
        action="require_2fa",
    ),
)

balance_resp = ResponseMessage(
    message_type=MessageType.BALANCE_UPDATE,
    response=BalanceUpdateResponse(
        account_id="RO12ABCD1234567890123456",
        old_balance=Decimal("10000.00"),
        new_balance=Decimal("5000.00"),
        currency=Currency.RON,
        transaction_id="TXN001",
    ),
)

profile_resp = ResponseMessage(
    message_type=MessageType.PROFILE_UPDATE,
    response=ProfileUpdateResponse(
        account_id="RO12ABCD1234567890123456",
        update_type=AccountUpdateType.CHANGE_NAME,
        status="success",
        details="Name updated successfully",
    ),
)

notify_resp = ResponseMessage(
    message_type=MessageType.NOTIFICATION,
    response=NotificationResponse(
        user_id="USER_456",
        channel="sms",
        message="Transfer de 5000 RON efectuat cu succes.",
        priority="normal",
    ),
)

if __name__ == "__main__":
    print("REQUEST MESSAGES (INPUT)")

    print("\n--- Transfer Request ---")
    print(transfer_req.model_dump_json(indent=2))

    print("\n--- Withdrawal Request ---")
    print(withdrawal_req.model_dump_json(indent=2))

    print("\n--- Deposit Request ---")
    print(deposit_req.model_dump_json(indent=2))

    print("\n--- Change Name Request ---")
    print(change_name_req.model_dump_json(indent=2))

    print("RESPONSE MESSAGES (OUTPUT)")

    print("\n--- Transaction Validated ---")
    print(validated_resp.model_dump_json(indent=2))

    print("\n--- Fraud Alert ---")
    print(fraud_resp.model_dump_json(indent=2))

    print("\n--- Balance Update ---")
    print(balance_resp.model_dump_json(indent=2))

    print("\n--- Profile Update ---")
    print(profile_resp.model_dump_json(indent=2))

    print("\n--- Notification ---")
    print(notify_resp.model_dump_json(indent=2))
