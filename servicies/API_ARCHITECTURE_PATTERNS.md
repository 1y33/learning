# API Architecture Patterns - Complete Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Layered Architecture (N-Tier)](#layered-architecture-n-tier)
3. [Clean Architecture](#clean-architecture)
4. [Hexagonal Architecture (Ports & Adapters)](#hexagonal-architecture-ports--adapters)
5. [Onion Architecture](#onion-architecture)
6. [CQRS Pattern](#cqrs-pattern)
7. [Event-Driven Architecture](#event-driven-architecture)
8. [Microservices Architecture](#microservices-architecture)
9. [Core Patterns Deep Dive](#core-patterns-deep-dive)
10. [SOLID Principles in Practice](#solid-principles-in-practice)
11. [Comparison & When to Use](#comparison--when-to-use)
12. [Complete Implementation Examples](#complete-implementation-examples)

---

## Introduction

### What is Software Architecture?

Software architecture defines the **high-level structure** of your application:
- How components are organized
- How they communicate
- How responsibilities are divided
- How data flows through the system

### Why Does Architecture Matter?

**Bad Architecture:**
```python
# Everything in one file - NIGHTMARE!
@app.route('/users', methods=['POST'])
def create_user():
    # Validation
    if not request.json.get('email'):
        return {"error": "Email required"}, 400

    # Business logic
    email = request.json['email']
    password = request.json['password']
    hashed_pw = bcrypt.hash(password)

    # Database access
    conn = psycopg2.connect("dbname=mydb")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users (email, password) VALUES (%s, %s)",
                   (email, hashed_pw))
    conn.commit()

    # Send email
    send_welcome_email(email)

    # Log
    logger.info(f"User created: {email}")

    return {"message": "User created"}, 201
```

**Good Architecture:**
```python
# Separated concerns - CLEAN!
@app.route('/users', methods=['POST'])
def create_user():
    dto = CreateUserDTO(**request.json)
    user = user_service.create_user(dto)
    return UserSchema().dump(user), 201
```

---

## Layered Architecture (N-Tier)

### Overview

The **Layered Architecture** (also called N-Tier) is the most traditional and widely used pattern. It organizes code into **horizontal layers**, where each layer has a specific responsibility.

### Structure

```
┌─────────────────────────────────────┐
│     Presentation Layer (API)        │  ← Controllers, Routes, Validation
├─────────────────────────────────────┤
│        Business Logic Layer         │  ← Services, Use Cases
├─────────────────────────────────────┤
│      Data Access Layer (DAL)        │  ← Repositories, ORM Models
├─────────────────────────────────────┤
│          Database Layer             │  ← PostgreSQL, MongoDB, etc.
└─────────────────────────────────────┘
```

### Key Principles

1. **Separation of Concerns** - Each layer has ONE responsibility
2. **Dependency Rule** - Upper layers can depend on lower layers, NOT vice versa
3. **Loose Coupling** - Layers communicate through interfaces

### Layers Explained

#### 1. Presentation Layer (Controllers)
**Responsibility:** Handle HTTP requests/responses, input validation, authentication

```python
# controllers/user_controller.py
from flask import Blueprint, request, jsonify
from services.user_service import UserService
from dto.user_dto import CreateUserDTO
from schemas.user_schema import UserSchema

user_bp = Blueprint('users', __name__)
user_service = UserService()

@user_bp.route('/users', methods=['POST'])
def create_user():
    """
    Presentation Layer:
    - Receives HTTP request
    - Validates input
    - Calls service layer
    - Returns HTTP response
    """
    try:
        # Parse and validate input
        dto = CreateUserDTO(**request.json)

        # Call business logic layer
        user = user_service.create_user(dto)

        # Serialize response
        return UserSchema().dump(user), 201

    except ValidationError as e:
        return {"error": str(e)}, 400
    except DuplicateEmailError as e:
        return {"error": str(e)}, 409
```

#### 2. Business Logic Layer (Services)
**Responsibility:** Core business rules, orchestration, transactions

```python
# services/user_service.py
from repositories.user_repository import UserRepository
from repositories.email_repository import EmailRepository
from utils.password import hash_password
from exceptions import DuplicateEmailError

class UserService:
    """
    Business Logic Layer:
    - Contains business rules
    - Orchestrates multiple repositories
    - Handles transactions
    """

    def __init__(self):
        self.user_repo = UserRepository()
        self.email_repo = EmailRepository()

    def create_user(self, dto: CreateUserDTO) -> User:
        """
        Business logic for creating a user.
        """
        # Business rule: Check if email exists
        if self.user_repo.find_by_email(dto.email):
            raise DuplicateEmailError("Email already exists")

        # Business rule: Password must be hashed
        hashed_password = hash_password(dto.password)

        # Create user entity
        user = User(
            email=dto.email,
            password=hashed_password,
            name=dto.name
        )

        # Save to database
        created_user = self.user_repo.create(user)

        # Send welcome email (another business rule)
        self.email_repo.send_welcome_email(created_user.email)

        return created_user

    def get_user_by_id(self, user_id: int) -> User:
        user = self.user_repo.find_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")
        return user

    def update_user(self, user_id: int, dto: UpdateUserDTO) -> User:
        user = self.get_user_by_id(user_id)

        # Business rule: Can't change email if already verified
        if user.email_verified and dto.email != user.email:
            raise ValidationError("Cannot change verified email")

        user.name = dto.name or user.name
        user.email = dto.email or user.email

        return self.user_repo.update(user)
```

#### 3. Data Access Layer (Repositories)
**Responsibility:** Database operations, queries, data persistence

```python
# repositories/user_repository.py
from sqlalchemy.orm import Session
from models.user import UserModel
from entities.user import User
from database import get_db

class UserRepository:
    """
    Data Access Layer:
    - Abstracts database operations
    - Converts between ORM models and domain entities
    - Handles queries
    """

    def __init__(self):
        self.db: Session = get_db()

    def create(self, user: User) -> User:
        """Create a new user in the database."""
        db_user = UserModel(
            email=user.email,
            password=user.password,
            name=user.name
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

        return self._to_entity(db_user)

    def find_by_id(self, user_id: int) -> User | None:
        """Find user by ID."""
        db_user = self.db.query(UserModel).filter(UserModel.id == user_id).first()
        return self._to_entity(db_user) if db_user else None

    def find_by_email(self, email: str) -> User | None:
        """Find user by email."""
        db_user = self.db.query(UserModel).filter(UserModel.email == email).first()
        return self._to_entity(db_user) if db_user else None

    def update(self, user: User) -> User:
        """Update existing user."""
        db_user = self.db.query(UserModel).filter(UserModel.id == user.id).first()
        if not db_user:
            raise UserNotFoundError()

        db_user.email = user.email
        db_user.name = user.name
        self.db.commit()
        self.db.refresh(db_user)

        return self._to_entity(db_user)

    def delete(self, user_id: int) -> bool:
        """Delete user by ID."""
        result = self.db.query(UserModel).filter(UserModel.id == user_id).delete()
        self.db.commit()
        return result > 0

    def find_all(self, skip: int = 0, limit: int = 100) -> list[User]:
        """Get all users with pagination."""
        db_users = self.db.query(UserModel).offset(skip).limit(limit).all()
        return [self._to_entity(u) for u in db_users]

    def _to_entity(self, db_user: UserModel) -> User:
        """Convert ORM model to domain entity."""
        return User(
            id=db_user.id,
            email=db_user.email,
            password=db_user.password,
            name=db_user.name,
            created_at=db_user.created_at
        )
```

### Project Structure

```
my_api/
├── app.py                      # Application entry point
├── config.py                   # Configuration
├── controllers/                # Presentation Layer
│   ├── __init__.py
│   ├── user_controller.py
│   └── product_controller.py
├── services/                   # Business Logic Layer
│   ├── __init__.py
│   ├── user_service.py
│   └── product_service.py
├── repositories/               # Data Access Layer
│   ├── __init__.py
│   ├── user_repository.py
│   └── product_repository.py
├── models/                     # ORM Models (SQLAlchemy)
│   ├── __init__.py
│   ├── user.py
│   └── product.py
├── entities/                   # Domain Entities
│   ├── __init__.py
│   ├── user.py
│   └── product.py
├── dto/                        # Data Transfer Objects
│   ├── __init__.py
│   ├── user_dto.py
│   └── product_dto.py
├── schemas/                    # Serialization (Marshmallow)
│   ├── __init__.py
│   ├── user_schema.py
│   └── product_schema.py
├── exceptions/                 # Custom Exceptions
│   └── __init__.py
├── utils/                      # Utilities
│   ├── password.py
│   └── validators.py
└── database.py                 # Database connection
```

### Advantages

✅ **Simple to understand** - Clear separation of concerns
✅ **Easy to maintain** - Each layer is independent
✅ **Testable** - Can test each layer separately
✅ **Team-friendly** - Different teams can work on different layers
✅ **Industry standard** - Most developers know this pattern

### Disadvantages

❌ **Database-centric** - Database is at the core
❌ **Tight coupling to frameworks** - Hard to switch databases/frameworks
❌ **Business logic can leak** - Sometimes spreads across layers
❌ **Testing requires database** - Integration tests are complex

### When to Use

✅ **CRUD applications** - Simple create/read/update/delete
✅ **Small to medium projects** - Not too complex
✅ **Rapid development** - Need to ship fast
✅ **Traditional web apps** - Not highly scalable systems

---

## Clean Architecture

### Overview

**Clean Architecture** (by Uncle Bob Martin) inverts the dependency direction. The **business logic (domain)** is at the center, and everything else depends on it.

### Key Principle: Dependency Inversion

```
Traditional (Layered):
Controller → Service → Repository → Database
(Inner layers depend on outer layers)

Clean Architecture:
Database → Repository Interface ← Service ← Controller
                ↑                    ↑
            (depends on)         (depends on)
```

### Structure

```
┌─────────────────────────────────────────────────┐
│         External Layer (Infrastructure)         │
│   ┌──────────────────────────────────────┐     │
│   │    Interface Adapters (Controllers)   │     │
│   │  ┌────────────────────────────────┐  │     │
│   │  │   Application Business Rules   │  │     │
│   │  │  ┌──────────────────────────┐  │  │     │
│   │  │  │   Enterprise Business    │  │  │     │
│   │  │  │        Rules (Entities)  │  │  │     │
│   │  │  └──────────────────────────┘  │  │     │
│   │  │         (Use Cases)            │  │     │
│   │  └────────────────────────────────┘  │     │
│   │      (Controllers, Presenters)       │     │
│   └──────────────────────────────────────┘     │
│     (DB, APIs, Frameworks, UI)                  │
└─────────────────────────────────────────────────┘
```

### Layers Explained

#### 1. Entities (Core Domain)
**Pure business logic - NO framework dependencies**

```python
# domain/entities/user.py
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class User:
    """
    Core Domain Entity - Pure business logic
    NO dependencies on frameworks, databases, or external libraries
    """
    id: Optional[int]
    email: str
    password: str
    name: str
    email_verified: bool = False
    created_at: Optional[datetime] = None

    def verify_email(self):
        """Business rule: Email verification"""
        if self.email_verified:
            raise ValueError("Email already verified")
        self.email_verified = True

    def can_change_email(self, new_email: str) -> bool:
        """Business rule: Can't change verified email"""
        if self.email_verified and new_email != self.email:
            return False
        return True

    def is_password_valid(self, password: str) -> bool:
        """Business rule: Password validation"""
        return len(password) >= 8
```

#### 2. Use Cases (Application Business Rules)
**Application-specific business rules**

```python
# application/use_cases/create_user_use_case.py
from domain.entities.user import User
from application.interfaces.user_repository import IUserRepository
from application.interfaces.email_service import IEmailService
from application.dto.create_user_dto import CreateUserDTO
from utils.password import hash_password

class CreateUserUseCase:
    """
    Use Case: Create a new user
    - Contains application-specific business logic
    - Depends on INTERFACES, not implementations
    """

    def __init__(
        self,
        user_repository: IUserRepository,  # ← Interface, not concrete class!
        email_service: IEmailService       # ← Interface, not concrete class!
    ):
        self.user_repository = user_repository
        self.email_service = email_service

    def execute(self, dto: CreateUserDTO) -> User:
        """Execute the use case."""

        # Business rule: Check if email exists
        existing_user = self.user_repository.find_by_email(dto.email)
        if existing_user:
            raise ValueError("Email already exists")

        # Create entity
        user = User(
            id=None,
            email=dto.email,
            password=hash_password(dto.password),
            name=dto.name
        )

        # Validate business rules
        if not user.is_password_valid(dto.password):
            raise ValueError("Password must be at least 8 characters")

        # Save user
        created_user = self.user_repository.save(user)

        # Send welcome email
        self.email_service.send_welcome_email(created_user.email)

        return created_user
```

#### 3. Interface Adapters (Controllers, Presenters, Gateways)

**Interfaces (Ports):**
```python
# application/interfaces/user_repository.py
from abc import ABC, abstractmethod
from domain.entities.user import User
from typing import Optional

class IUserRepository(ABC):
    """
    Repository Interface (Port)
    - Defines contract for data access
    - Implementation is in infrastructure layer
    """

    @abstractmethod
    def save(self, user: User) -> User:
        pass

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    def update(self, user: User) -> User:
        pass

    @abstractmethod
    def delete(self, user_id: int) -> bool:
        pass
```

**Controller:**
```python
# infrastructure/web/controllers/user_controller.py
from flask import Blueprint, request, jsonify
from application.use_cases.create_user_use_case import CreateUserUseCase
from application.dto.create_user_dto import CreateUserDTO
from infrastructure.di_container import get_container

user_bp = Blueprint('users', __name__)

@user_bp.route('/users', methods=['POST'])
def create_user():
    """
    Controller (Interface Adapter)
    - Converts HTTP request to DTO
    - Calls use case
    - Converts result to HTTP response
    """
    container = get_container()
    use_case = container.resolve(CreateUserUseCase)

    try:
        dto = CreateUserDTO(**request.json)
        user = use_case.execute(dto)

        return jsonify({
            "id": user.id,
            "email": user.email,
            "name": user.name
        }), 201

    except ValueError as e:
        return {"error": str(e)}, 400
```

#### 4. Infrastructure (External Layer)

**Repository Implementation:**
```python
# infrastructure/persistence/user_repository_impl.py
from sqlalchemy.orm import Session
from application.interfaces.user_repository import IUserRepository
from domain.entities.user import User
from infrastructure.persistence.models.user_model import UserModel

class UserRepositoryImpl(IUserRepository):
    """
    Concrete Repository Implementation
    - Implements the interface defined in application layer
    - Contains framework-specific code (SQLAlchemy)
    """

    def __init__(self, db: Session):
        self.db = db

    def save(self, user: User) -> User:
        db_user = UserModel(
            email=user.email,
            password=user.password,
            name=user.name,
            email_verified=user.email_verified
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

        return self._to_entity(db_user)

    def find_by_email(self, email: str) -> User | None:
        db_user = self.db.query(UserModel).filter(UserModel.email == email).first()
        return self._to_entity(db_user) if db_user else None

    def _to_entity(self, db_user: UserModel) -> User:
        return User(
            id=db_user.id,
            email=db_user.email,
            password=db_user.password,
            name=db_user.name,
            email_verified=db_user.email_verified,
            created_at=db_user.created_at
        )
```

### Dependency Injection Container

```python
# infrastructure/di_container.py
from dependency_injector import containers, providers
from infrastructure.persistence.user_repository_impl import UserRepositoryImpl
from infrastructure.email.email_service_impl import EmailServiceImpl
from application.use_cases.create_user_use_case import CreateUserUseCase
from database import get_db

class Container(containers.DeclarativeContainer):
    """Dependency Injection Container"""

    # Database
    db = providers.Singleton(get_db)

    # Repositories (Infrastructure → Interface)
    user_repository = providers.Factory(
        UserRepositoryImpl,
        db=db
    )

    # Services
    email_service = providers.Factory(EmailServiceImpl)

    # Use Cases
    create_user_use_case = providers.Factory(
        CreateUserUseCase,
        user_repository=user_repository,
        email_service=email_service
    )
```

### Project Structure

```
my_api/
├── domain/                     # Core Business Logic (Entities)
│   ├── entities/
│   │   ├── user.py
│   │   └── product.py
│   └── value_objects/
│       └── email.py
├── application/                # Application Business Rules
│   ├── use_cases/
│   │   ├── create_user_use_case.py
│   │   ├── get_user_use_case.py
│   │   └── update_user_use_case.py
│   ├── interfaces/             # Ports (Abstractions)
│   │   ├── user_repository.py
│   │   └── email_service.py
│   └── dto/
│       └── create_user_dto.py
├── infrastructure/             # External Layer
│   ├── web/                    # Web Framework
│   │   ├── controllers/
│   │   │   └── user_controller.py
│   │   └── app.py
│   ├── persistence/            # Database
│   │   ├── models/
│   │   │   └── user_model.py
│   │   ├── user_repository_impl.py
│   │   └── database.py
│   ├── email/                  # Email Service
│   │   └── email_service_impl.py
│   └── di_container.py         # Dependency Injection
├── tests/
│   ├── unit/
│   └── integration/
└── main.py
```

### Advantages

✅ **Framework independent** - Can swap Flask → FastAPI easily
✅ **Database independent** - Can swap PostgreSQL → MongoDB easily
✅ **Testable** - Core logic has NO external dependencies
✅ **Business logic first** - Domain is protected
✅ **Long-term maintainability** - Scales well with complexity

### Disadvantages

❌ **More complex** - Requires understanding of abstractions
❌ **More boilerplate** - More files and interfaces
❌ **Steeper learning curve** - Not intuitive for beginners
❌ **Overkill for simple apps** - Too much for CRUD apps

### When to Use

✅ **Complex business logic** - Domain rules are important
✅ **Long-term projects** - Will be maintained for years
✅ **Need to change frameworks** - Want flexibility
✅ **High testability required** - Banking, healthcare, fintech
✅ **Enterprise applications** - Large teams, complex requirements

---

## Hexagonal Architecture (Ports & Adapters)

### Overview

**Hexagonal Architecture** (by Alistair Cockburn) is similar to Clean Architecture but focuses on **isolating the application core** from external concerns using **Ports** (interfaces) and **Adapters** (implementations).

### Key Concept: Ports & Adapters

```
        ┌────────────────────────────────┐
        │         Adapters               │
        │  ┌──────────────────────────┐  │
        │  │       Ports              │  │
        │  │  ┌────────────────────┐  │  │
        │  │  │  Application Core  │  │  │
        │  │  │   (Domain Logic)   │  │  │
        │  │  └────────────────────┘  │  │
        │  │         Ports            │  │
        │  └──────────────────────────┘  │
        │         Adapters               │
        └────────────────────────────────┘

Ports: Interfaces (what the app needs)
Adapters: Implementations (how it's done)
```

### Ports (Interfaces)

**Primary Ports (Driving)** - What drives your application (HTTP, CLI, gRPC)
**Secondary Ports (Driven)** - What your application drives (Database, Email, Cache)

```python
# ports/input/user_service_port.py
from abc import ABC, abstractmethod
from domain.user import User

class UserServicePort(ABC):
    """
    Primary Port (Driving Port)
    - Defines operations the application provides
    - Driven by external actors (controllers)
    """

    @abstractmethod
    def create_user(self, email: str, password: str, name: str) -> User:
        pass

    @abstractmethod
    def get_user(self, user_id: int) -> User:
        pass
```

```python
# ports/output/user_repository_port.py
from abc import ABC, abstractmethod
from domain.user import User

class UserRepositoryPort(ABC):
    """
    Secondary Port (Driven Port)
    - Defines operations the application needs from infrastructure
    - Driven by the application core
    """

    @abstractmethod
    def save(self, user: User) -> User:
        pass

    @abstractmethod
    def find_by_id(self, user_id: int) -> User | None:
        pass
```

### Application Core (Domain)

```python
# core/user_service.py
from ports.input.user_service_port import UserServicePort
from ports.output.user_repository_port import UserRepositoryPort
from domain.user import User
from utils.password import hash_password

class UserService(UserServicePort):
    """
    Application Core
    - Implements business logic
    - Depends ONLY on ports (interfaces)
    - NO dependencies on frameworks or infrastructure
    """

    def __init__(self, user_repository: UserRepositoryPort):
        self.user_repository = user_repository

    def create_user(self, email: str, password: str, name: str) -> User:
        # Business logic
        existing = self.user_repository.find_by_email(email)
        if existing:
            raise ValueError("Email exists")

        user = User(
            id=None,
            email=email,
            password=hash_password(password),
            name=name
        )

        return self.user_repository.save(user)

    def get_user(self, user_id: int) -> User:
        user = self.user_repository.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")
        return user
```

### Adapters

**Primary Adapter (HTTP Controller):**
```python
# adapters/input/http/user_controller.py
from flask import Blueprint, request, jsonify
from ports.input.user_service_port import UserServicePort

class UserController:
    """
    Primary Adapter (HTTP)
    - Adapts HTTP requests to application core
    - Depends on PRIMARY PORT (UserServicePort)
    """

    def __init__(self, user_service: UserServicePort):
        self.user_service = user_service
        self.blueprint = Blueprint('users', __name__)
        self._register_routes()

    def _register_routes(self):
        self.blueprint.add_url_rule(
            '/users',
            'create_user',
            self.create_user,
            methods=['POST']
        )

    def create_user(self):
        data = request.json
        user = self.user_service.create_user(
            email=data['email'],
            password=data['password'],
            name=data['name']
        )
        return jsonify({"id": user.id, "email": user.email}), 201
```

**Secondary Adapter (PostgreSQL Repository):**
```python
# adapters/output/persistence/postgres_user_repository.py
from sqlalchemy.orm import Session
from ports.output.user_repository_port import UserRepositoryPort
from domain.user import User
from adapters.output.persistence.models import UserModel

class PostgresUserRepository(UserRepositoryPort):
    """
    Secondary Adapter (PostgreSQL)
    - Implements SECONDARY PORT (UserRepositoryPort)
    - Contains database-specific code
    """

    def __init__(self, db: Session):
        self.db = db

    def save(self, user: User) -> User:
        db_user = UserModel(
            email=user.email,
            password=user.password,
            name=user.name
        )
        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

        user.id = db_user.id
        return user

    def find_by_id(self, user_id: int) -> User | None:
        db_user = self.db.query(UserModel).filter(UserModel.id == user_id).first()
        if not db_user:
            return None

        return User(
            id=db_user.id,
            email=db_user.email,
            password=db_user.password,
            name=db_user.name
        )
```

### Project Structure

```
my_api/
├── domain/                         # Core Domain
│   ├── user.py
│   └── product.py
├── ports/                          # Interfaces
│   ├── input/                      # Primary Ports (driving)
│   │   └── user_service_port.py
│   └── output/                     # Secondary Ports (driven)
│       ├── user_repository_port.py
│       └── email_service_port.py
├── core/                           # Application Core
│   ├── user_service.py
│   └── product_service.py
├── adapters/                       # Implementations
│   ├── input/                      # Primary Adapters
│   │   ├── http/                   # HTTP Adapter
│   │   │   └── user_controller.py
│   │   └── cli/                    # CLI Adapter
│   │       └── user_cli.py
│   └── output/                     # Secondary Adapters
│       ├── persistence/            # Database Adapter
│       │   ├── postgres_user_repository.py
│       │   └── models/
│       ├── email/                  # Email Adapter
│       │   └── smtp_email_service.py
│       └── cache/                  # Cache Adapter
│           └── redis_cache.py
├── config/
│   └── di_container.py
└── main.py
```

### Advantages

✅ **Extremely flexible** - Swap adapters easily
✅ **Testable** - Mock adapters in tests
✅ **Technology agnostic** - Core doesn't know about tech stack
✅ **Multiple interfaces** - HTTP + CLI + gRPC simultaneously
✅ **Clear boundaries** - Ports define clear contracts

### Disadvantages

❌ **Complex** - Many interfaces and adapters
❌ **Over-engineering** - Too much for simple apps
❌ **Verbose** - Lots of boilerplate code

### When to Use

✅ **Multiple interfaces** - Need HTTP + CLI + gRPC
✅ **Changing requirements** - Might swap databases/frameworks
✅ **Complex integrations** - Many external systems
✅ **Long-term projects** - Will evolve over years

---

## Onion Architecture

### Overview

**Onion Architecture** (by Jeffrey Palermo) is similar to Clean Architecture but emphasizes **concentric layers** where inner layers never depend on outer layers.

### Structure

```
        ┌───────────────────────────────┐
        │   Infrastructure Layer        │
        │  ┌─────────────────────────┐  │
        │  │  Application Services   │  │
        │  │  ┌───────────────────┐  │  │
        │  │  │  Domain Services  │  │  │
        │  │  │  ┌─────────────┐  │  │  │
        │  │  │  │   Domain    │  │  │  │
        │  │  │  │   Model     │  │  │  │
        │  │  │  └─────────────┘  │  │  │
        │  │  └───────────────────┘  │  │
        │  └─────────────────────────┘  │
        └───────────────────────────────┘

Dependencies flow INWARD only!
```

### Layers

1. **Domain Model (Center)** - Entities, Value Objects
2. **Domain Services** - Business rules operating on entities
3. **Application Services** - Use cases, orchestration
4. **Infrastructure** - Frameworks, databases, external services

### Key Implementation

```python
# 1. Domain Model (Center)
# domain/entities/user.py
@dataclass
class User:
    id: int
    email: str
    password: str
    name: str

    def change_password(self, old_password: str, new_password: str):
        if not self.verify_password(old_password):
            raise ValueError("Invalid password")
        self.password = hash_password(new_password)

# 2. Domain Services
# domain/services/user_domain_service.py
class UserDomainService:
    """Business rules that involve multiple entities"""

    def can_user_access_resource(self, user: User, resource: Resource) -> bool:
        # Complex business logic
        return user.role == 'admin' or resource.owner_id == user.id

# 3. Application Services
# application/services/user_application_service.py
class UserApplicationService:
    """Orchestrates use cases"""

    def __init__(self, user_repo: IUserRepository):
        self.user_repo = user_repo
        self.domain_service = UserDomainService()

    def register_user(self, dto: RegisterUserDTO) -> User:
        user = User(...)
        return self.user_repo.save(user)

# 4. Infrastructure
# infrastructure/persistence/user_repository.py
class UserRepository(IUserRepository):
    """Database implementation"""
    pass
```

### Advantages

✅ **Clear dependencies** - Always inward
✅ **Domain-centric** - Business logic is protected
✅ **Testable** - Core has no dependencies

### Disadvantages

❌ **Abstract** - Harder to understand initially
❌ **More layers** - Can be overkill

### When to Use

✅ **Domain-driven design** - Rich domain models
✅ **Complex business rules** - Many domain services
✅ **Enterprise applications**

---

## CQRS Pattern

### Overview

**CQRS** (Command Query Responsibility Segregation) **separates read and write operations** into different models.

### Core Principle

```
Traditional:
  UserService
      ↓
  UserRepository
      ↓
  Single Database

CQRS:
  Commands (Write)    Queries (Read)
      ↓                   ↓
  Write Model        Read Model
      ↓                   ↓
  Write DB           Read DB (denormalized)
```

### Commands (Write)

```python
# commands/create_user_command.py
from dataclasses import dataclass

@dataclass
class CreateUserCommand:
    """Command: Intent to create a user"""
    email: str
    password: str
    name: str

# commands/handlers/create_user_handler.py
class CreateUserCommandHandler:
    """Handles the create user command"""

    def __init__(self, write_repository: UserWriteRepository, event_bus: EventBus):
        self.write_repo = write_repository
        self.event_bus = event_bus

    def handle(self, command: CreateUserCommand) -> int:
        # Create user in write model
        user = User(
            email=command.email,
            password=hash_password(command.password),
            name=command.name
        )

        user_id = self.write_repo.save(user)

        # Publish event for read model to consume
        self.event_bus.publish(UserCreatedEvent(
            user_id=user_id,
            email=command.email,
            name=command.name
        ))

        return user_id
```

### Queries (Read)

```python
# queries/get_user_query.py
@dataclass
class GetUserQuery:
    """Query: Request to get user data"""
    user_id: int

# queries/handlers/get_user_handler.py
class GetUserQueryHandler:
    """Handles the get user query"""

    def __init__(self, read_repository: UserReadRepository):
        self.read_repo = read_repository

    def handle(self, query: GetUserQuery) -> UserDTO:
        # Read from optimized read model
        return self.read_repo.get_user_by_id(query.user_id)
```

### Read Model Sync (Event Handler)

```python
# event_handlers/user_created_event_handler.py
class UserCreatedEventHandler:
    """Updates read model when user is created"""

    def __init__(self, read_repository: UserReadRepository):
        self.read_repo = read_repository

    def handle(self, event: UserCreatedEvent):
        # Update denormalized read model
        self.read_repo.create_user_view(
            user_id=event.user_id,
            email=event.email,
            name=event.name,
            # Add computed/denormalized data
            display_name=f"{event.name} ({event.email})"
        )
```

### Project Structure

```
my_api/
├── commands/                   # Write side
│   ├── create_user_command.py
│   └── handlers/
│       └── create_user_handler.py
├── queries/                    # Read side
│   ├── get_user_query.py
│   └── handlers/
│       └── get_user_handler.py
├── domain/                     # Domain model (write side)
│   └── user.py
├── read_models/                # Denormalized read models
│   └── user_view.py
├── events/                     # Domain events
│   └── user_created_event.py
├── event_handlers/             # Event handlers (sync read model)
│   └── user_created_event_handler.py
└── infrastructure/
    ├── write_repository.py     # Write database
    └── read_repository.py      # Read database
```

### Advantages

✅ **Performance** - Optimize read and write separately
✅ **Scalability** - Scale reads independently from writes
✅ **Flexibility** - Different databases for read/write
✅ **Read optimization** - Denormalized, fast queries

### Disadvantages

❌ **Complexity** - Two models to maintain
❌ **Eventual consistency** - Read model may be stale
❌ **More infrastructure** - Needs event bus, multiple DBs

### When to Use

✅ **Read-heavy systems** - 90% reads, 10% writes
✅ **Complex queries** - Need denormalized data
✅ **High scalability** - Need to scale reads independently
✅ **Event-driven** - Already using event sourcing

❌ **Simple CRUD** - Overkill for basic apps

---

## Event-Driven Architecture

### Overview

**Event-Driven Architecture (EDA)** uses **events** as the primary mechanism for communication between services.

### Key Components

1. **Events** - Something that happened
2. **Event Producers** - Publish events
3. **Event Consumers** - Subscribe to events
4. **Event Bus** - Routes events (RabbitMQ, Kafka, etc.)

### Implementation

**Event:**
```python
# events/user_registered_event.py
from dataclasses import dataclass
from datetime import datetime

@dataclass
class UserRegisteredEvent:
    """Event: A user was registered"""
    event_id: str
    user_id: int
    email: str
    name: str
    timestamp: datetime
```

**Event Publisher:**
```python
# services/user_service.py
class UserService:
    def __init__(self, user_repo: UserRepository, event_bus: EventBus):
        self.user_repo = user_repo
        self.event_bus = event_bus

    def register_user(self, dto: RegisterUserDTO) -> User:
        # Create user
        user = User(...)
        created_user = self.user_repo.save(user)

        # Publish event
        event = UserRegisteredEvent(
            event_id=str(uuid.uuid4()),
            user_id=created_user.id,
            email=created_user.email,
            name=created_user.name,
            timestamp=datetime.now()
        )
        self.event_bus.publish('user.registered', event)

        return created_user
```

**Event Consumers:**
```python
# event_consumers/send_welcome_email_consumer.py
class SendWelcomeEmailConsumer:
    def __init__(self, email_service: EmailService):
        self.email_service = email_service

    def handle(self, event: UserRegisteredEvent):
        self.email_service.send_welcome_email(
            to=event.email,
            name=event.name
        )

# event_consumers/create_user_profile_consumer.py
class CreateUserProfileConsumer:
    def __init__(self, profile_service: ProfileService):
        self.profile_service = profile_service

    def handle(self, event: UserRegisteredEvent):
        self.profile_service.create_default_profile(event.user_id)
```

**Event Bus (RabbitMQ):**
```python
# infrastructure/event_bus/rabbitmq_event_bus.py
import pika
import json

class RabbitMQEventBus:
    def __init__(self, host='localhost'):
        self.connection = pika.BlockingConnection(pika.ConnectionParameters(host))
        self.channel = self.connection.channel()

    def publish(self, event_type: str, event: any):
        self.channel.exchange_declare(exchange='events', exchange_type='topic')

        message = json.dumps(event.__dict__, default=str)
        self.channel.basic_publish(
            exchange='events',
            routing_key=event_type,
            body=message
        )

    def subscribe(self, event_type: str, handler):
        self.channel.exchange_declare(exchange='events', exchange_type='topic')

        result = self.channel.queue_declare(queue='', exclusive=True)
        queue_name = result.method.queue

        self.channel.queue_bind(
            exchange='events',
            queue=queue_name,
            routing_key=event_type
        )

        def callback(ch, method, properties, body):
            event_data = json.loads(body)
            handler(event_data)

        self.channel.basic_consume(
            queue=queue_name,
            on_message_callback=callback,
            auto_ack=True
        )

        self.channel.start_consuming()
```

### Advantages

✅ **Loose coupling** - Services don't know about each other
✅ **Scalability** - Easy to add new consumers
✅ **Resilience** - If one service fails, others continue
✅ **Flexibility** - Easy to add new features

### Disadvantages

❌ **Complexity** - Requires message broker
❌ **Debugging** - Harder to trace flow
❌ **Eventual consistency** - Data may be stale

### When to Use

✅ **Microservices** - Services need to communicate
✅ **Asynchronous workflows** - Don't need immediate response
✅ **High scalability** - Need to scale independently
✅ **Complex workflows** - Multiple steps/services

---

## Microservices Architecture

### Overview

**Microservices** split your application into **small, independent services** that communicate over a network.

### Characteristics

1. **Single Responsibility** - Each service does ONE thing
2. **Independent Deployment** - Deploy services separately
3. **Decentralized Data** - Each service has its own database
4. **Communication** - REST, gRPC, message queues

### Example: E-commerce System

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  User Service   │    │ Product Service │    │  Order Service  │
│                 │    │                 │    │                 │
│  - Register     │    │  - CRUD         │    │  - Create Order │
│  - Login        │    │  - Inventory    │    │  - Get Orders   │
│  - Profile      │    │  - Search       │    │  - Cancel Order │
│                 │    │                 │    │                 │
│  PostgreSQL     │    │   MongoDB       │    │   PostgreSQL    │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                       │
         └──────────────────────┼───────────────────────┘
                                │
                        ┌───────┴────────┐
                        │   API Gateway  │
                        │   (Kong/Nginx) │
                        └────────────────┘
```

### Service Communication

**1. Synchronous (REST):**
```python
# order_service/services/order_service.py
import requests

class OrderService:
    def create_order(self, user_id: int, product_id: int, quantity: int):
        # Call User Service
        user_response = requests.get(f"http://user-service/api/users/{user_id}")
        if user_response.status_code != 200:
            raise ValueError("User not found")

        # Call Product Service
        product_response = requests.get(f"http://product-service/api/products/{product_id}")
        if product_response.status_code != 200:
            raise ValueError("Product not found")

        product = product_response.json()

        # Create order
        order = Order(
            user_id=user_id,
            product_id=product_id,
            quantity=quantity,
            total_price=product['price'] * quantity
        )

        return self.order_repo.save(order)
```

**2. Asynchronous (Message Queue):**
```python
# order_service/services/order_service.py
class OrderService:
    def __init__(self, order_repo, event_bus):
        self.order_repo = order_repo
        self.event_bus = event_bus

    def create_order(self, user_id: int, product_id: int, quantity: int):
        order = Order(...)
        created_order = self.order_repo.save(order)

        # Publish event - other services can react
        self.event_bus.publish('order.created', {
            'order_id': created_order.id,
            'user_id': user_id,
            'product_id': product_id,
            'quantity': quantity
        })

        return created_order

# inventory_service/consumers/order_created_consumer.py
class OrderCreatedConsumer:
    def handle(self, event):
        # Reduce inventory
        self.inventory_service.reduce_stock(
            product_id=event['product_id'],
            quantity=event['quantity']
        )

# notification_service/consumers/order_created_consumer.py
class OrderCreatedConsumer:
    def handle(self, event):
        # Send notification
        self.notification_service.send_order_confirmation(
            user_id=event['user_id'],
            order_id=event['order_id']
        )
```

### Service Structure (Each Microservice)

```
user-service/
├── api/
│   └── controllers/
│       └── user_controller.py
├── services/
│   └── user_service.py
├── repositories/
│   └── user_repository.py
├── models/
│   └── user.py
├── config/
│   └── database.py
├── Dockerfile
├── requirements.txt
└── main.py

product-service/
├── api/
├── services/
├── repositories/
├── models/
├── Dockerfile
└── main.py

order-service/
├── api/
├── services/
├── repositories/
├── models/
├── Dockerfile
└── main.py
```

### Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  user-service:
    build: ./user-service
    ports:
      - "5001:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@user-db/userdb
    depends_on:
      - user-db

  user-db:
    image: postgres:14
    environment:
      POSTGRES_DB: userdb
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass

  product-service:
    build: ./product-service
    ports:
      - "5002:5000"
    environment:
      - MONGO_URL=mongodb://product-db:27017/productdb
    depends_on:
      - product-db

  product-db:
    image: mongo:5

  order-service:
    build: ./order-service
    ports:
      - "5003:5000"
    environment:
      - DATABASE_URL=postgresql://user:pass@order-db/orderdb
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - order-db
      - rabbitmq

  order-db:
    image: postgres:14

  rabbitmq:
    image: rabbitmq:3-management
    ports:
      - "5672:5672"
      - "15672:15672"

  api-gateway:
    image: kong:latest
    ports:
      - "8000:8000"
      - "8001:8001"
```

### Advantages

✅ **Independent deployment** - Deploy services separately
✅ **Technology diversity** - Use different languages/databases per service
✅ **Scalability** - Scale only what you need
✅ **Resilience** - One service failure doesn't bring down everything
✅ **Team autonomy** - Teams own services independently

### Disadvantages

❌ **Complexity** - Distributed system challenges
❌ **Network latency** - Service-to-service calls are slow
❌ **Data consistency** - Hard to maintain consistency across services
❌ **Testing** - End-to-end testing is complex
❌ **Deployment** - Requires orchestration (Kubernetes)
❌ **Monitoring** - Need distributed tracing (Jaeger)

### When to Use

✅ **Large teams** - Multiple teams working independently
✅ **Complex applications** - Many bounded contexts
✅ **High scalability** - Need to scale different parts independently
✅ **Technology diversity** - Want to use different tech stacks

❌ **Small teams** - Too much overhead
❌ **Simple applications** - Start with monolith

---

## Core Patterns Deep Dive

### Repository Pattern

**Purpose:** Abstract data access layer

```python
# repositories/user_repository.py
from abc import ABC, abstractmethod
from typing import List, Optional
from entities.user import User

class IUserRepository(ABC):
    """Repository interface - abstraction over data access"""

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[User]:
        pass

    @abstractmethod
    def find_by_email(self, email: str) -> Optional[User]:
        pass

    @abstractmethod
    def find_all(self, skip: int = 0, limit: int = 100) -> List[User]:
        pass

    @abstractmethod
    def save(self, user: User) -> User:
        pass

    @abstractmethod
    def update(self, user: User) -> User:
        pass

    @abstractmethod
    def delete(self, user_id: int) -> bool:
        pass

# PostgreSQL implementation
class PostgresUserRepository(IUserRepository):
    def __init__(self, db: Session):
        self.db = db

    def find_by_id(self, user_id: int) -> Optional[User]:
        db_user = self.db.query(UserModel).filter(UserModel.id == user_id).first()
        return self._to_entity(db_user) if db_user else None

    def save(self, user: User) -> User:
        db_user = UserModel(**user.__dict__)
        self.db.add(db_user)
        self.db.commit()
        return self._to_entity(db_user)

# MongoDB implementation
class MongoUserRepository(IUserRepository):
    def __init__(self, db):
        self.collection = db['users']

    def find_by_id(self, user_id: int) -> Optional[User]:
        doc = self.collection.find_one({"_id": user_id})
        return User(**doc) if doc else None

    def save(self, user: User) -> User:
        result = self.collection.insert_one(user.__dict__)
        user.id = result.inserted_id
        return user
```

**Benefits:**
- ✅ Database agnostic
- ✅ Easy to test (mock repository)
- ✅ Centralized data access logic

---

### Service Layer Pattern

**Purpose:** Encapsulate business logic

```python
# services/user_service.py
from repositories.user_repository import IUserRepository
from repositories.email_repository import IEmailRepository
from utils.password import hash_password, verify_password

class UserService:
    """Service layer - business logic orchestration"""

    def __init__(
        self,
        user_repository: IUserRepository,
        email_repository: IEmailRepository
    ):
        self.user_repo = user_repository
        self.email_repo = email_repository

    def register_user(self, email: str, password: str, name: str) -> User:
        """
        Business logic for user registration:
        1. Validate email doesn't exist
        2. Hash password
        3. Create user
        4. Send welcome email
        """
        # Business rule: Email must be unique
        if self.user_repo.find_by_email(email):
            raise DuplicateEmailError("Email already exists")

        # Business rule: Password must be hashed
        hashed_pw = hash_password(password)

        # Create user
        user = User(
            id=None,
            email=email,
            password=hashed_pw,
            name=name
        )

        created_user = self.user_repo.save(user)

        # Business rule: Send welcome email
        self.email_repo.send_welcome_email(created_user.email)

        return created_user

    def authenticate_user(self, email: str, password: str) -> User:
        """Business logic for authentication"""
        user = self.user_repo.find_by_email(email)

        if not user:
            raise AuthenticationError("Invalid credentials")

        if not verify_password(password, user.password):
            raise AuthenticationError("Invalid credentials")

        return user

    def change_password(self, user_id: int, old_password: str, new_password: str):
        """Business logic for changing password"""
        user = self.user_repo.find_by_id(user_id)

        if not user:
            raise UserNotFoundError()

        # Business rule: Must verify old password
        if not verify_password(old_password, user.password):
            raise AuthenticationError("Invalid old password")

        # Business rule: New password must be different
        if old_password == new_password:
            raise ValidationError("New password must be different")

        user.password = hash_password(new_password)
        self.user_repo.update(user)
```

**Benefits:**
- ✅ Centralized business logic
- ✅ Reusable across controllers
- ✅ Easier to test business rules
- ✅ Transaction management

---

### DTO (Data Transfer Object) Pattern

**Purpose:** Transfer data between layers without exposing internal models

```python
# dto/create_user_dto.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class CreateUserDTO:
    """
    DTO: Data Transfer Object
    - Used for input validation
    - Decouples API from domain model
    """
    email: str
    password: str
    name: str
    age: Optional[int] = None

    def validate(self):
        """Validate input"""
        if not self.email or '@' not in self.email:
            raise ValidationError("Invalid email")

        if not self.password or len(self.password) < 8:
            raise ValidationError("Password must be at least 8 characters")

        if not self.name:
            raise ValidationError("Name is required")

@dataclass
class UpdateUserDTO:
    name: Optional[str] = None
    age: Optional[int] = None

@dataclass
class UserResponseDTO:
    """DTO for API responses - doesn't include password!"""
    id: int
    email: str
    name: str
    created_at: str

# Usage in controller
@app.route('/users', methods=['POST'])
def create_user():
    # Parse DTO
    dto = CreateUserDTO(**request.json)
    dto.validate()

    # Call service
    user = user_service.create_user(dto)

    # Convert to response DTO
    response_dto = UserResponseDTO(
        id=user.id,
        email=user.email,
        name=user.name,
        created_at=user.created_at.isoformat()
    )

    return jsonify(asdict(response_dto)), 201
```

**Benefits:**
- ✅ Decouples API from domain
- ✅ Input validation
- ✅ Security (don't expose internal fields)
- ✅ API versioning

---

### Dependency Injection

**Purpose:** Invert dependencies, easier testing

**Without DI (BAD):**
```python
class UserService:
    def __init__(self):
        # Hard-coded dependencies - BAD!
        self.user_repo = PostgresUserRepository(get_db())
        self.email_service = SMTPEmailService('smtp.gmail.com')

    def create_user(self, dto):
        # Can't test this easily!
        pass
```

**With DI (GOOD):**
```python
class UserService:
    def __init__(
        self,
        user_repository: IUserRepository,      # ← Interface, not concrete class
        email_service: IEmailService           # ← Interface, not concrete class
    ):
        self.user_repo = user_repository
        self.email_service = email_service

    def create_user(self, dto):
        # Easy to test - inject mocks!
        pass

# Production usage
user_service = UserService(
    user_repository=PostgresUserRepository(db),
    email_service=SMTPEmailService('smtp.gmail.com')
)

# Test usage
user_service = UserService(
    user_repository=MockUserRepository(),     # ← Mock
    email_service=MockEmailService()          # ← Mock
)
```

**Using Dependency Injector library:**
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    # Config
    config = providers.Configuration()

    # Database
    db = providers.Singleton(Database, db_url=config.db_url)

    # Repositories
    user_repository = providers.Factory(
        PostgresUserRepository,
        db=db
    )

    # Services
    email_service = providers.Factory(
        SMTPEmailService,
        smtp_host=config.smtp_host
    )

    user_service = providers.Factory(
        UserService,
        user_repository=user_repository,
        email_service=email_service
    )

# Usage
container = Container()
container.config.from_dict({
    'db_url': 'postgresql://localhost/db',
    'smtp_host': 'smtp.gmail.com'
})

user_service = container.user_service()
```

---

### Unit of Work Pattern

**Purpose:** Maintain a list of objects affected by a transaction and coordinates changes

```python
# unit_of_work.py
from sqlalchemy.orm import Session

class UnitOfWork:
    """
    Unit of Work: Coordinates multiple repository operations in a single transaction
    """

    def __init__(self, session: Session):
        self.session = session
        self.user_repository = None
        self.product_repository = None
        self.order_repository = None

    def __enter__(self):
        self.user_repository = UserRepository(self.session)
        self.product_repository = ProductRepository(self.session)
        self.order_repository = OrderRepository(self.session)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.rollback()
        self.session.close()

    def commit(self):
        self.session.commit()

    def rollback(self):
        self.session.rollback()

# Usage
def create_order_with_uow(user_id: int, product_id: int, quantity: int):
    with UnitOfWork(get_db()) as uow:
        # Get user
        user = uow.user_repository.find_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        # Get product
        product = uow.product_repository.find_by_id(product_id)
        if not product:
            raise ValueError("Product not found")

        # Check stock
        if product.stock < quantity:
            raise ValueError("Insufficient stock")

        # Create order
        order = Order(user_id=user_id, product_id=product_id, quantity=quantity)
        uow.order_repository.save(order)

        # Reduce stock
        product.stock -= quantity
        uow.product_repository.update(product)

        # Commit all changes in one transaction
        uow.commit()

        return order
```

**Benefits:**
- ✅ All-or-nothing transactions
- ✅ Maintains consistency
- ✅ Centralized transaction management

---

## SOLID Principles in Practice

### S - Single Responsibility Principle

**Definition:** A class should have ONE reason to change

**BAD:**
```python
class User:
    def __init__(self, email, password):
        self.email = email
        self.password = password

    def save_to_database(self):
        # Database logic - WRONG!
        pass

    def send_email(self):
        # Email logic - WRONG!
        pass

    def validate(self):
        # Validation logic - MAYBE OK
        pass
```

**GOOD:**
```python
class User:
    """ONE responsibility: Represent user data and business rules"""
    def __init__(self, email, password):
        self.email = email
        self.password = password

    def validate(self):
        if '@' not in self.email:
            raise ValueError("Invalid email")

class UserRepository:
    """ONE responsibility: Database operations"""
    def save(self, user: User):
        # Database logic
        pass

class EmailService:
    """ONE responsibility: Send emails"""
    def send_welcome_email(self, email: str):
        # Email logic
        pass
```

---

### O - Open/Closed Principle

**Definition:** Open for extension, closed for modification

**BAD:**
```python
class PaymentProcessor:
    def process_payment(self, payment_type, amount):
        if payment_type == 'credit_card':
            # Process credit card
            pass
        elif payment_type == 'paypal':
            # Process PayPal
            pass
        elif payment_type == 'bitcoin':  # ← Need to modify class!
            # Process Bitcoin
            pass
```

**GOOD:**
```python
from abc import ABC, abstractmethod

class PaymentMethod(ABC):
    @abstractmethod
    def process(self, amount: float):
        pass

class CreditCardPayment(PaymentMethod):
    def process(self, amount: float):
        # Credit card logic
        pass

class PayPalPayment(PaymentMethod):
    def process(self, amount: float):
        # PayPal logic
        pass

class BitcoinPayment(PaymentMethod):  # ← Just add new class!
    def process(self, amount: float):
        # Bitcoin logic
        pass

class PaymentProcessor:
    def process_payment(self, payment_method: PaymentMethod, amount: float):
        payment_method.process(amount)  # ← No modification needed!
```

---

### L - Liskov Substitution Principle

**Definition:** Subtypes must be substitutable for their base types

**BAD:**
```python
class Bird:
    def fly(self):
        print("Flying")

class Penguin(Bird):
    def fly(self):
        raise Exception("Penguins can't fly!")  # ← Violates LSP!
```

**GOOD:**
```python
class Bird:
    def move(self):
        pass

class FlyingBird(Bird):
    def move(self):
        self.fly()

    def fly(self):
        print("Flying")

class Penguin(Bird):
    def move(self):
        self.swim()

    def swim(self):
        print("Swimming")
```

---

### I - Interface Segregation Principle

**Definition:** Don't force clients to depend on interfaces they don't use

**BAD:**
```python
class IWorker(ABC):
    @abstractmethod
    def work(self):
        pass

    @abstractmethod
    def eat(self):
        pass

class Robot(IWorker):
    def work(self):
        print("Working")

    def eat(self):
        # Robots don't eat! - VIOLATES ISP
        raise NotImplementedError()
```

**GOOD:**
```python
class IWorkable(ABC):
    @abstractmethod
    def work(self):
        pass

class IEatable(ABC):
    @abstractmethod
    def eat(self):
        pass

class Human(IWorkable, IEatable):
    def work(self):
        print("Working")

    def eat(self):
        print("Eating")

class Robot(IWorkable):  # ← Only implements what it needs
    def work(self):
        print("Working")
```

---

### D - Dependency Inversion Principle

**Definition:** Depend on abstractions, not concretions

**BAD:**
```python
class MySQLDatabase:
    def save(self, data):
        # MySQL-specific code
        pass

class UserService:
    def __init__(self):
        self.db = MySQLDatabase()  # ← Depends on concrete class!

    def create_user(self, user):
        self.db.save(user)
```

**GOOD:**
```python
class IDatabase(ABC):
    @abstractmethod
    def save(self, data):
        pass

class MySQLDatabase(IDatabase):
    def save(self, data):
        # MySQL-specific code
        pass

class PostgreSQLDatabase(IDatabase):
    def save(self, data):
        # PostgreSQL-specific code
        pass

class UserService:
    def __init__(self, database: IDatabase):  # ← Depends on abstraction!
        self.db = database

    def create_user(self, user):
        self.db.save(user)
```

---

## Comparison & When to Use

### Architecture Comparison Table

| Architecture | Complexity | Testability | Flexibility | Best For |
|--------------|------------|-------------|-------------|----------|
| **Layered (N-Tier)** | ⭐ Low | ⭐⭐ Medium | ⭐⭐ Medium | CRUD apps, MVPs, small projects |
| **Clean Architecture** | ⭐⭐⭐ High | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | Enterprise, long-term projects |
| **Hexagonal** | ⭐⭐⭐⭐ Very High | ⭐⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Excellent | Multiple interfaces (HTTP+CLI+gRPC) |
| **Onion** | ⭐⭐⭐ High | ⭐⭐⭐⭐ High | ⭐⭐⭐⭐ High | Domain-driven design |
| **CQRS** | ⭐⭐⭐⭐ Very High | ⭐⭐⭐ Medium | ⭐⭐⭐⭐ High | Read-heavy systems, event sourcing |
| **Event-Driven** | ⭐⭐⭐ High | ⭐⭐⭐ Medium | ⭐⭐⭐⭐⭐ Excellent | Microservices, async workflows |
| **Microservices** | ⭐⭐⭐⭐⭐ Very High | ⭐⭐ Low | ⭐⭐⭐⭐⭐ Excellent | Large teams, high scalability |

### Decision Tree

```
Start Here
    │
    ├─ Simple CRUD app?
    │       └─ YES → Layered Architecture (N-Tier)
    │
    ├─ Complex business logic?
    │       └─ YES → Clean Architecture or Onion Architecture
    │
    ├─ Need multiple interfaces (HTTP + CLI + gRPC)?
    │       └─ YES → Hexagonal Architecture
    │
    ├─ Read-heavy with complex queries?
    │       └─ YES → CQRS
    │
    ├─ Need asynchronous communication?
    │       └─ YES → Event-Driven Architecture
    │
    └─ Large team, need independent services?
            └─ YES → Microservices
```

### Project Size Guide

**Small Project (1-3 devs, <10k lines)**
- ✅ Layered Architecture
- ❌ Avoid: Microservices, Hexagonal

**Medium Project (3-10 devs, 10k-100k lines)**
- ✅ Clean Architecture
- ✅ CQRS (if read-heavy)
- ✅ Event-Driven (if async needed)
- ❌ Avoid: Microservices (unless necessary)

**Large Project (10+ devs, 100k+ lines)**
- ✅ Clean/Hexagonal/Onion Architecture
- ✅ CQRS + Event Sourcing
- ✅ Microservices (with API Gateway)
- ✅ Event-Driven Architecture

---

## Complete Implementation Examples

### Example 1: E-commerce API (Layered Architecture)

```python
# models/product.py (ORM Model)
from sqlalchemy import Column, Integer, String, Float
from database import Base

class ProductModel(Base):
    __tablename__ = 'products'

    id = Column(Integer, primary_key=True)
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    stock = Column(Integer, nullable=False)

# entities/product.py (Domain Entity)
from dataclasses import dataclass

@dataclass
class Product:
    id: int
    name: str
    price: float
    stock: int

# dto/product_dto.py
@dataclass
class CreateProductDTO:
    name: str
    price: float
    stock: int

# repositories/product_repository.py
class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def find_by_id(self, product_id: int) -> Product | None:
        db_product = self.db.query(ProductModel).filter(ProductModel.id == product_id).first()
        if not db_product:
            return None
        return Product(
            id=db_product.id,
            name=db_product.name,
            price=db_product.price,
            stock=db_product.stock
        )

    def save(self, product: Product) -> Product:
        db_product = ProductModel(
            name=product.name,
            price=product.price,
            stock=product.stock
        )
        self.db.add(db_product)
        self.db.commit()
        self.db.refresh(db_product)
        product.id = db_product.id
        return product

    def update_stock(self, product_id: int, quantity: int):
        db_product = self.db.query(ProductModel).filter(ProductModel.id == product_id).first()
        if db_product:
            db_product.stock -= quantity
            self.db.commit()

# services/product_service.py
class ProductService:
    def __init__(self, product_repo: ProductRepository):
        self.product_repo = product_repo

    def create_product(self, dto: CreateProductDTO) -> Product:
        product = Product(
            id=None,
            name=dto.name,
            price=dto.price,
            stock=dto.stock
        )
        return self.product_repo.save(product)

    def get_product(self, product_id: int) -> Product:
        product = self.product_repo.find_by_id(product_id)
        if not product:
            raise ProductNotFoundError(f"Product {product_id} not found")
        return product

    def reduce_stock(self, product_id: int, quantity: int):
        product = self.get_product(product_id)

        if product.stock < quantity:
            raise InsufficientStockError(f"Only {product.stock} items available")

        self.product_repo.update_stock(product_id, quantity)

# controllers/product_controller.py
from flask import Blueprint, request, jsonify

product_bp = Blueprint('products', __name__)
product_service = ProductService(ProductRepository(get_db()))

@product_bp.route('/products', methods=['POST'])
def create_product():
    dto = CreateProductDTO(**request.json)
    product = product_service.create_product(dto)
    return jsonify({
        "id": product.id,
        "name": product.name,
        "price": product.price,
        "stock": product.stock
    }), 201

@product_bp.route('/products/<int:product_id>', methods=['GET'])
def get_product(product_id):
    try:
        product = product_service.get_product(product_id)
        return jsonify({
            "id": product.id,
            "name": product.name,
            "price": product.price,
            "stock": product.stock
        })
    except ProductNotFoundError as e:
        return {"error": str(e)}, 404
```

---

## Key Takeaways

### Start Simple, Evolve

1. **Start:** Layered Architecture (Controller-Service-Repository)
2. **Evolve:** Add Clean Architecture patterns when needed
3. **Scale:** Microservices only when you have multiple teams

### Core Principles

✅ **Separation of Concerns** - Each component does ONE thing
✅ **Dependency Inversion** - Depend on abstractions, not concretions
✅ **Testability** - Design for testing from day one
✅ **SOLID Principles** - Follow religiously

### Common Mistakes to Avoid

❌ **Over-engineering** - Don't use microservices for a small app
❌ **Under-engineering** - Don't put everything in one file
❌ **Skipping tests** - Write tests for business logic
❌ **Ignoring SOLID** - Will cause pain later
❌ **Database-first design** - Start with domain, not database

### Learning Path

1. Master **Layered Architecture** first
2. Learn **SOLID principles**
3. Understand **Clean Architecture**
4. Explore **CQRS** and **Event-Driven**
5. Finally, **Microservices** (when you really need them)

---

**Remember:** The best architecture is the one that solves YOUR problem with the right balance of simplicity and scalability! 🚀
