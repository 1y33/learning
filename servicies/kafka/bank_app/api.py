from typing import Optional, Literal
from pydantic import Field, BaseModel, field_validator
from datetime import datetime
from decimal import Decimal
from enum import Enum
import re

### MAIN ENUMS

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

class TransactionRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1)
    amount: Decimal = Field(..., gt=0, decimal_places=2)
    currency: Currency = Currency.RON
    metadata: RequestMetadata

    @field_validator("amount")
    @classmethod
    def validate_amount(cls, v: Decimal) -> Decimal:
        if v <= 0:
            raise ValueError("Amount must be positive")
        if v > Decimal("1000000"):
            raise ValueError("Amount exceeds maximum limit")
        return v

class TransferRequest(TransactionRequest):
    to_account: str = Field(..., min_length=5, max_length=34)
    beneficiary_name: Optional[str] = Field(None, max_length=100)

class WithdrawalRequest(TransactionRequest):
    atm_id: Optional[str] = None

class DepositRequest(TransactionRequest):
    source: Literal["cash_counter", "check", "wire_transfer"] = "cash_counter"


### Updates Requests
 
class AccountUpdateRequest(BaseModel):
    account_id: str = Field(..., min_length=5, max_length=34)
    user_id: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.now)

class ChangeNameRequest(AccountUpdateRequest):
    old_name: str = Field(..., min_length=2, max_length=100)
    new_name: str = Field(..., min_length=2, max_length=100)

    @field_validator("new_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not re.match(r"^[a-zA-ZăâîșțĂÂÎȘȚ\s\-\.]+$", v):
            raise ValueError("Name contains invalid characters")
        return v.strip()

class CreateCardRequest(AccountUpdateRequest):
    card_type: CardType = CardType.DEBIT
    card_name: str = Field(..., max_length=50)
    virtual_card: bool = False

class CloseAccountRequest(AccountUpdateRequest):
    reason: str = Field(default="user_request", max_length=200)
    final_balance: Decimal = Field(..., ge=0)
    transfer_to_account: Optional[str] = None
    
    
    
    
    
