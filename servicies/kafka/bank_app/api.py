from typing import Optional, Literal, Union
from pydantic import Field, BaseModel, field_validator
from datetime import datetime
from decimal import Decimal
from enum import Enum
import re

### MAIN ENUMS


class MessageType(str, Enum):
    TRANSACTION_REQUEST = "transaction_request"
    ACCOUNT_UPDATE_REQUEST = "account_update_request"
    TRANSACTION_VALIDATED = "transaction_validated"
    FRAUD_ALERT = "fraud_alert"
    BALANCE_UPDATE = "balance_update"
    PROFILE_UPDATE = "profile_update"
    NOTIFICATION = "notification"


class TransactionType(str, Enum):
    TRANSFER = "transfer"
    WITHDRAWAL = "withdrawal"
    DEPOSIT = "deposit"


class AccountUpdateType(str, Enum):
    CHANGE_NAME = "change_name"
    CREATE_CARD = "create_card"
    CLOSE_ACCOUNT = "close_account"


class Currency(str, Enum):
    RON = "RON"
    EUR = "EUR"
    USD = "USD"


class CardType(str, Enum):
    DEBIT = "debit"
    CREDIT = "credit"


##### REQUEST API


class RequestMetadata(BaseModel):
    account_id: str = Field(..., min_length=5, max_length=34)
    timestamp: datetime = Field(default_factory=datetime.now)
    user_ip: Optional[str] = None
    device_id: Optional[str] = None
    user_agent: Optional[str] = None
    location: Optional[str] = None

    @field_validator("account_id")
    @classmethod
    def validate_account_id(cls, v: str) -> str:
        if v.startswith("RO") and not re.match(r"^RO\d{2}[A-Z]{4}\d{16}$", v):
            raise ValueError("Invalid Romanian IBAN format")
        return v


#### Transaction API


class TransferData(BaseModel):
    to_account: str = Field(..., min_length=5, max_length=34)
    beneficiary_name: Optional[str] = Field(None, max_length=100)


class WithdrawalData(BaseModel):
    atm_id: Optional[str] = None


class DepositData(BaseModel):
    source: Literal["cash_counter", "check", "wire_transfer"] = "cash_counter"


class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    transaction_type: TransactionType
    amount: Decimal = Field(..., gt=0, decimal_places=2)
    currency: Currency = Currency.RON
    metadata: RequestMetadata
    transfer: Optional[TransferData] = None
    withdrawal: Optional[WithdrawalData] = None
    deposit: Optional[DepositData] = None

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > Decimal("1000000"):
            raise ValueError("Amount exceeds maximum limit")
        return v


### Updates Requests


class ChangeNameData(BaseModel):
    old_name: str = Field(..., min_length=2, max_length=100)
    new_name: str = Field(..., min_length=2, max_length=100)

    @field_validator("new_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-ZăâîșțĂÂÎȘȚ\s\-\.]+$", v):
            raise ValueError("Name contains invalid characters")
        return v.strip()


class CreateCardData(BaseModel):
    card_type: CardType = CardType.DEBIT
    card_name: str = Field(..., max_length=50)
    virtual_card: bool = False


class CloseAccountData(BaseModel):
    reason: str = Field(default="user_request", max_length=200)
    final_balance: Decimal = Field(..., ge=0)
    transfer_to_account: Optional[str] = None


class AccountUpdateRequest(BaseModel):
    account_id: str = Field(..., min_length=5, max_length=34)
    user_id: str = Field(..., min_length=1)
    update_type: AccountUpdateType
    timestamp: datetime = Field(default_factory=datetime.now)
    change_name: Optional[ChangeNameData] = None
    create_card: Optional[CreateCardData] = None
    close_account: Optional[CloseAccountData] = None


##### RESPONSE API


class TransactionValidatedResponse(BaseModel):
    transaction_id: str
    status: Literal["approved", "rejected", "pending"]
    reference_number: str
    processed_at: datetime = Field(default_factory=datetime.now)


class FraudAlertResponse(BaseModel):
    transaction_id: str
    risk_score: float = Field(..., ge=0, le=1)
    risk_level: Literal["low", "medium", "high", "critical"]
    reasons: list[str]
    action: Literal["allow", "require_2fa", "block", "review"]


class BalanceUpdateResponse(BaseModel):
    account_id: str
    old_balance: Decimal
    new_balance: Decimal
    currency: Currency
    transaction_id: str


class ProfileUpdateResponse(BaseModel):
    account_id: str
    update_type: AccountUpdateType
    status: Literal["success", "failed", "pending"]
    details: Optional[str] = None


class NotificationResponse(BaseModel):
    user_id: str
    channel: Literal["sms", "email", "push"]
    message: str
    priority: Literal["low", "normal", "high"] = "normal"


##### MESSAGE WRAPPERS


class RequestMessage(BaseModel):
    message_type: MessageType
    request: Union[TransactionRequest, AccountUpdateRequest]


class ResponseMessage(BaseModel):
    message_type: MessageType
    response: Union[
        TransactionValidatedResponse,
        FraudAlertResponse,
        BalanceUpdateResponse,
        ProfileUpdateResponse,
        NotificationResponse
    ]
