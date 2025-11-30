This is a showcase of having a modualr approach of Consumer - Producer so we are not limited by having a 1 to 1 process . Now we are getting different requests and we have a consumer for the bank and for the producer we have a way to add topics



```json
REQUEST MESSAGES (INPUT)

--- Transfer Request ---
{
  "message_type": "transaction_request",
  "request": {
    "transaction_id": "TXN001",
    "transaction_type": "transfer",
    "amount": "5000.00",
    "currency": "RON",
    "metadata": {
      "account_id": "RO12ABCD1234567890123456",
      "timestamp": "2025-11-30T17:36:06.322304",
      "user_ip": "192.168.1.100",
      "device_id": "mobile_app_v2",
      "user_agent": "BankApp/2.0 Android",
      "location": "Bucharest, RO"
    },
    "transfer": {
      "to_account": "RO89DEFG9876543210123456",
      "beneficiary_name": "Maria Ionescu"
    },
    "withdrawal": null,
    "deposit": null
  }
}

--- Withdrawal Request ---
{
  "message_type": "transaction_request",
  "request": {
    "transaction_id": "TXN002",
    "transaction_type": "withdrawal",
    "amount": "2000.00",
    "currency": "RON",
    "metadata": {
      "account_id": "RO12ABCD1234567890123456",
      "timestamp": "2025-11-30T17:36:06.322304",
      "user_ip": "192.168.1.100",
      "device_id": "mobile_app_v2",
      "user_agent": "BankApp/2.0 Android",
      "location": "Bucharest, RO"
    },
    "transfer": null,
    "withdrawal": {
      "atm_id": "ATM_CENTER_001"
    },
    "deposit": null
  }
}

--- Deposit Request ---
{
  "message_type": "transaction_request",
  "request": {
    "transaction_id": "TXN003",
    "transaction_type": "deposit",
    "amount": "10000.00",
    "currency": "EUR",
    "metadata": {
      "account_id": "RO12ABCD1234567890123456",
      "timestamp": "2025-11-30T17:36:06.322304",
      "user_ip": "192.168.1.100",
      "device_id": "mobile_app_v2",
      "user_agent": "BankApp/2.0 Android",
      "location": "Bucharest, RO"
    },
    "transfer": null,
    "withdrawal": null,
    "deposit": {
      "source": "cash_counter"
    }
  }
}

--- Change Name Request ---
{
  "message_type": "account_update_request",
  "request": {
    "account_id": "RO12ABCD1234567890123456",
    "user_id": "USER_456",
    "update_type": "change_name",
    "timestamp": "2025-11-30T17:36:06.322526",
    "change_name": {
      "old_name": "Ion Popescu",
      "new_name": "Ion P. Popescu"
    },
    "create_card": null,
    "close_account": null
  }
}
RESPONSE MESSAGES (OUTPUT)

--- Transaction Validated ---
{
  "message_type": "transaction_validated",
  "response": {
    "transaction_id": "TXN001",
    "status": "approved",
    "reference_number": "REF2025113000001",
    "processed_at": "2025-11-30T17:36:06.322620"
  }
}

--- Fraud Alert ---
{
  "message_type": "fraud_alert",
  "response": {
    "transaction_id": "TXN001",
    "risk_score": 0.85,
    "risk_level": "high",
    "reasons": [
      "Amount exceeds daily average",
      "New beneficiary"
    ],
    "action": "require_2fa"
  }
}

--- Balance Update ---
{
  "message_type": "balance_update",
  "response": {
    "account_id": "RO12ABCD1234567890123456",
    "old_balance": "10000.00",
    "new_balance": "5000.00",
    "currency": "RON",
    "transaction_id": "TXN001"
  }
}

--- Profile Update ---
{
  "message_type": "profile_update",
  "response": {
    "account_id": "RO12ABCD1234567890123456",
    "update_type": "change_name",
    "status": "success",
    "details": "Name updated successfully"
  }
}

--- Notification ---
{
  "message_type": "notification",
  "response": {
    "user_id": "USER_456",
    "channel": "sms",
    "message": "Transfer de 5000 RON efectuat cu succes.",
    "priority": "normal"
  }
}
```