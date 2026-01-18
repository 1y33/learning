# Advanced Flask Tricks & Best Practices

## Table of Contents
1. [Configuration Management](#configuration-management)
2. [Database Integration](#database-integration)
3. [Authentication & Security](#authentication--security)
4. [Middleware & Hooks](#middleware--hooks)
5. [Caching](#caching)
6. [Rate Limiting](#rate-limiting)
7. [Error Handling Advanced](#error-handling-advanced)
8. [Logging & Monitoring](#logging--monitoring)
9. [Testing](#testing)
10. [Production Tips](#production-tips)

---

## Configuration Management

### ✅ Use Config Classes

```python
# config.py
import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key'
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class DevelopmentConfig(Config):
    DEBUG = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///dev.db'

class ProductionConfig(Config):
    DEBUG = False
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL')

class TestingConfig(Config):
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///test.db'

# app.py
app.config.from_object('config.DevelopmentConfig')
```

### ✅ Environment Variables (.env)

```bash
# .env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-super-secret-key
DATABASE_URL=postgresql://user:pass@localhost/db
```

```python
# app.py
from dotenv import load_dotenv
import os

load_dotenv()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
```

---

## Database Integration

### ✅ SQLAlchemy ORM

```python
from flask_sqlalchemy import SQLAlchemy

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'
db = SQLAlchemy(app)

# Model
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)

    def to_dict(self):
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email
        }

# Usage
@app.route('/users', methods=['POST'])
def create_user():
    user = User(
        username=request.json['username'],
        email=request.json['email']
    )
    db.session.add(user)
    db.session.commit()
    return jsonify(user.to_dict()), 201

@app.route('/users')
def get_users():
    users = User.query.all()
    return jsonify([u.to_dict() for u in users])
```

### ✅ Database Migrations (Flask-Migrate)

```bash
pip install flask-migrate

flask db init
flask db migrate -m "Initial migration"
flask db upgrade
```

```python
from flask_migrate import Migrate

db = SQLAlchemy(app)
migrate = Migrate(app, db)
```

---

## Authentication & Security

### ✅ JWT Authentication

```python
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity

app.config['JWT_SECRET_KEY'] = 'super-secret-key'
jwt = JWTManager(app)

# Login
@app.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')

    # Check credentials (use bcrypt in production!)
    if username == 'admin' and password == 'password':
        access_token = create_access_token(identity=username)
        return jsonify(access_token=access_token)

    return jsonify({"error": "Invalid credentials"}), 401

# Protected route
@app.route('/protected', methods=['GET'])
@jwt_required()
def protected():
    current_user = get_jwt_identity()
    return jsonify(logged_in_as=current_user)
```

### ✅ Password Hashing (bcrypt)

```python
from flask_bcrypt import Bcrypt

bcrypt = Bcrypt(app)

# Hash password
hashed_pw = bcrypt.generate_password_hash('my_password').decode('utf-8')

# Check password
bcrypt.check_password_hash(hashed_pw, 'my_password')  # True
```

### ✅ CORS (Cross-Origin Resource Sharing)

```python
from flask_cors import CORS

# Allow all origins
CORS(app)

# Or specific configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://myapp.com"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

---

## Middleware & Hooks

### ✅ Before/After Request Hooks

```python
@app.before_request
def before_request():
    """Runs BEFORE each request"""
    g.start_time = time.time()
    print(f"Request started: {request.method} {request.path}")

@app.after_request
def after_request(response):
    """Runs AFTER each request"""
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        print(f"Request took {elapsed:.2f}s")

    # Add headers
    response.headers['X-API-Version'] = '1.0'
    return response

@app.teardown_request
def teardown_request(exception=None):
    """Cleanup after request (even if error occurred)"""
    if exception:
        print(f"Error occurred: {exception}")
```

### ✅ Context Variables (g object)

```python
from flask import g
import sqlite3

@app.before_request
def before_request():
    g.db = sqlite3.connect('database.db')

@app.teardown_request
def teardown_request(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()
```

### ✅ Custom Decorators

```python
from functools import wraps

# Decorator for admin verification
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_admin:
            return jsonify({"error": "Admin only"}), 403
        return f(*args, **kwargs)
    return decorated_function

@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    return jsonify({"message": "Welcome admin"})
```

---

## Caching

### ✅ Flask-Caching

```python
from flask_caching import Cache

cache = Cache(app, config={
    'CACHE_TYPE': 'simple',  # sau 'redis', 'memcached'
    'CACHE_DEFAULT_TIMEOUT': 300
})

# Cache for 5 minutes
@app.route('/expensive')
@cache.cached(timeout=300)
def expensive_operation():
    # Expensive operation
    result = heavy_computation()
    return jsonify(result)

# Cache with parameters
@app.route('/users/<int:user_id>')
@cache.cached(timeout=60, query_string=True)
def get_user(user_id):
    return jsonify({"user_id": user_id})

# Clear cache manually
@app.route('/clear-cache')
def clear_cache():
    cache.clear()
    return jsonify({"message": "Cache cleared"})
```

---

## Rate Limiting

### ✅ Flask-Limiter

```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Limit specific endpoint
@app.route('/api/expensive')
@limiter.limit("5 per minute")
def expensive_endpoint():
    return jsonify({"data": "..."})

# Exempt from rate limiting
@app.route('/health')
@limiter.exempt
def health():
    return jsonify({"status": "ok"})
```

---

## Error Handling Advanced

### ✅ Custom Error Pages & Handlers

```python
from werkzeug.exceptions import HTTPException

# JSON error handler pentru toate erorile
@app.errorhandler(HTTPException)
def handle_exception(e):
    response = {
        "error": e.name,
        "message": e.description,
        "status": e.code
    }
    return jsonify(response), e.code

# Custom exception
class ValidationError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({"error": error.message}), error.status_code

# Usage
@app.route('/validate')
def validate():
    if not valid_data:
        raise ValidationError("Data invalid")
    return jsonify({"status": "ok"})
```

### ✅ Structured Error Responses

```python
def error_response(message, status_code, errors=None):
    payload = {
        'success': False,
        'message': message,
        'status': status_code
    }
    if errors:
        payload['errors'] = errors
    return jsonify(payload), status_code

@app.route('/users', methods=['POST'])
def create_user():
    errors = []
    if not request.json.get('email'):
        errors.append({'field': 'email', 'message': 'Email is required'})
    if not request.json.get('password'):
        errors.append({'field': 'password', 'message': 'Password is required'})

    if errors:
        return error_response('Validation failed', 400, errors)

    return jsonify({"success": True}), 201
```

---

## Logging & Monitoring

### ✅ Advanced Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
if not app.debug:
    file_handler = RotatingFileHandler(
        'logs/app.log',
        maxBytes=10240000,  # 10MB
        backupCount=10
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('App startup')

# Usage
@app.route('/process')
def process_data():
    app.logger.info('Processing started')
    try:
        result = do_something()
        app.logger.info('Processing completed')
        return jsonify(result)
    except Exception as e:
        app.logger.error(f'Processing failed: {str(e)}')
        return jsonify({"error": "Processing failed"}), 500
```

### ✅ Request Logging Middleware

```python
import time
from flask import request, g

@app.before_request
def log_request():
    g.start_time = time.time()

@app.after_request
def log_response(response):
    if hasattr(g, 'start_time'):
        elapsed = time.time() - g.start_time
        app.logger.info(
            f'{request.method} {request.path} '
            f'{response.status_code} {elapsed:.2f}s'
        )
    return response
```

---

## Testing

### ✅ Unit Testing

```python
# test_app.py
import unittest
from app import app, db

class APITestCase(unittest.TestCase):
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test.db'
        self.app = app.test_client()
        db.create_all()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

    def test_get_users(self):
        response = self.app.get('/api/users')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn('users', data)

    def test_create_user(self):
        response = self.app.post('/api/users', json={
            'name': 'Test User',
            'email': 'test@example.com'
        })
        self.assertEqual(response.status_code, 201)

if __name__ == '__main__':
    unittest.main()
```

### ✅ Pytest

```python
# conftest.py
import pytest
from app import create_app, db

@pytest.fixture
def app():
    app = create_app('testing')
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    return app.test_client()

# test_users.py
def test_get_users(client):
    response = client.get('/api/users')
    assert response.status_code == 200
    assert 'users' in response.json
```

---

## Production Tips

### ✅ Application Factory Pattern

```python
# app/__init__.py
from flask import Flask
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

def create_app(config_name='development'):
    app = Flask(__name__)
    app.config.from_object(f'config.{config_name.capitalize()}Config')

    db.init_app(app)

    # Register blueprints
    from app.routes.users import users_bp
    from app.routes.products import products_bp
    app.register_blueprint(users_bp)
    app.register_blueprint(products_bp)

    return app

# run.py
from app import create_app

app = create_app('production')

if __name__ == '__main__':
    app.run()
```

### ✅ Structured Project Layout

```
my_api/
├── app/
│   ├── __init__.py          # Application factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py
│   │   └── product.py
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── users.py         # Users blueprint
│   │   └── products.py      # Products blueprint
│   ├── services/            # Business logic
│   │   ├── __init__.py
│   │   └── user_service.py
│   └── utils/               # Helper functions
│       ├── __init__.py
│       └── validators.py
├── tests/
│   ├── __init__.py
│   ├── test_users.py
│   └── test_products.py
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── .env                     # Environment variables
└── run.py                   # Entry point
```

### ✅ Use Gunicorn for Production

```bash
pip install gunicorn

# Run with 4 workers
gunicorn -w 4 -b 0.0.0.0:8000 app:app

# With config file
gunicorn -c gunicorn_config.py app:app
```

```python
# gunicorn_config.py
bind = "0.0.0.0:8000"
workers = 4
worker_class = "sync"
timeout = 120
keepalive = 5
errorlog = "logs/gunicorn_error.log"
accesslog = "logs/gunicorn_access.log"
```

### ✅ Use WSGI Server (uWSGI)

```bash
pip install uwsgi

uwsgi --http :8000 --wsgi-file app.py --callable app --processes 4 --threads 2
```

### ✅ Nginx Reverse Proxy

```nginx
# /etc/nginx/sites-available/myapi
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

---

## Performance Tips

### ✅ Database Query Optimization

```python
# ❌ BAD: N+1 queries
users = User.query.all()
for user in users:
    print(user.posts)  # Separate query for each user!

# ✅ GOOD: Eager loading
users = User.query.options(db.joinedload(User.posts)).all()
for user in users:
    print(user.posts)  # Already loaded!
```

### ✅ Pagination

```python
from flask import request

@app.route('/users')
def get_users():
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)

    pagination = User.query.paginate(
        page=page,
        per_page=per_page,
        error_out=False
    )

    return jsonify({
        'users': [u.to_dict() for u in pagination.items],
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })
```

### ✅ Async Support (Flask 2.0+)

```python
import asyncio

@app.route('/async')
async def async_endpoint():
    result = await async_operation()
    return jsonify(result)
```

---

## Security Best Practices

### ✅ Validate Input

```python
from flask_inputs import Inputs
from wtforms import validators

class UserInputs(Inputs):
    json = {
        'email': [validators.InputRequired(), validators.Email()],
        'age': [validators.InputRequired(), validators.NumberRange(min=18)]
    }

@app.route('/users', methods=['POST'])
def create_user():
    inputs = UserInputs(request)
    if not inputs.validate():
        return jsonify({"errors": inputs.errors}), 400
    # Process valid data
```

### ✅ SQL Injection Prevention

```python
# ❌ NEVER do this
query = f"SELECT * FROM users WHERE id = {user_id}"

# ✅ Always use parameterized queries (ORM does this automatically)
user = User.query.filter_by(id=user_id).first()

# Or with raw SQL
db.session.execute("SELECT * FROM users WHERE id = :id", {"id": user_id})
```

### ✅ Environment-based Settings

```python
# Never hardcode secrets!
# ❌ BAD
SECRET_KEY = 'my-secret-123'

# ✅ GOOD
SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
```

---

## Quick Tips Checklist

- ✅ Use blueprints for organization
- ✅ Implement proper error handling
- ✅ Add logging everywhere
- ✅ Use environment variables
- ✅ Implement rate limiting
- ✅ Cache expensive operations
- ✅ Paginate large datasets
- ✅ Use ORM (SQLAlchemy)
- ✅ Hash passwords (bcrypt)
- ✅ Add CORS if needed
- ✅ Write tests
- ✅ Use Application Factory pattern
- ✅ Run with Gunicorn/uWSGI in production
- ✅ Never commit secrets to Git
- ✅ Document your API

---

## Useful Extensions

| Extension | Purpose |
|-----------|---------|
| Flask-SQLAlchemy | Database ORM |
| Flask-Migrate | Database migrations |
| Flask-CORS | Cross-Origin Resource Sharing |
| Flask-JWT-Extended | JWT authentication |
| Flask-Bcrypt | Password hashing |
| Flask-Caching | Caching support |
| Flask-Limiter | Rate limiting |
| Flask-Admin | Admin interface |
| Flask-RESTful | REST API building |
| Flask-Marshmallow | Serialization/deserialization |

---

## Resources

- [Flask Documentation](https://flask.palletsprojects.com/)
- [Flask Mega Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [Real Python Flask Tutorials](https://realpython.com/tutorials/flask/)
- [Awesome Flask](https://github.com/humiaozuzu/awesome-flask)

---

**Remember**: Start simple, add complexity as needed! 🚀
