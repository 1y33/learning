# Tutorial Flask - Crearea unui API REST

## Cuprins
1. [Ce este Flask?](#ce-este-flask)
2. [Instalare și Setup](#instalare-și-setup)
3. [Concepte de Bază](#concepte-de-bază)
4. [Primul API](#primul-api)
5. [HTTP Methods](#http-methods)
6. [Routing și URL Parameters](#routing-și-url-parameters)
7. [Request și Response](#request-și-response)
8. [CRUD Operations](#crud-operations)
9. [Error Handling](#error-handling)
10. [Best Practices](#best-practices)

---

## Ce este Flask?

Flask este un **micro-framework** web pentru Python. Este:
- **Simplu** - ușor de învățat și folosit
- **Flexibil** - nu impune o structură rigidă
- **Extensibil** - multe extensii disponibile
- **Perfect pentru API-uri** - ideal pentru REST APIs

---

## Instalare și Setup

### 1. Instalare Flask

```bash
# Creează un virtual environment (recomandat)
python -m venv venv

# Activează virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate

# Instalează Flask
pip install flask

# Instalează dependențe utile
pip install flask-cors  # Pentru CORS
pip install python-dotenv  # Pentru variabile de mediu
```

### 2. Structura proiectului

```
my_api/
├── app.py              # Aplicația principală
├── config.py           # Configurări
├── requirements.txt    # Dependențe
├── .env               # Variabile de mediu
└── venv/              # Virtual environment
```

---

## Concepte de Bază

### Aplicația Flask

```python
from flask import Flask

# Creează instanța aplicației
app = Flask(__name__)

# Rulează aplicația
if __name__ == '__main__':
    app.run(debug=True)
```

### Decoratori și Rute

```python
@app.route('/hello')  # Decoratorul definește ruta
def hello():
    return "Hello, World!"
```

---

## Primul API

```python
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({
        "message": "Bun venit la API-ul meu!",
        "version": "1.0"
    })

@app.route('/api/status')
def status():
    return jsonify({
        "status": "online",
        "timestamp": "2024-01-18"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

**Rulare:**
```bash
python app.py
# Accesează: http://localhost:5000
```

---

## HTTP Methods

Flask suportă toate metodele HTTP:

```python
from flask import request

# GET - Citire date
@app.route('/items', methods=['GET'])
def get_items():
    return jsonify({"items": []})

# POST - Creare
@app.route('/items', methods=['POST'])
def create_item():
    data = request.json
    return jsonify({"created": data}), 201

# PUT - Actualizare completă
@app.route('/items/<int:id>', methods=['PUT'])
def update_item(id):
    data = request.json
    return jsonify({"updated": id})

# PATCH - Actualizare parțială
@app.route('/items/<int:id>', methods=['PATCH'])
def patch_item(id):
    data = request.json
    return jsonify({"patched": id})

# DELETE - Ștergere
@app.route('/items/<int:id>', methods=['DELETE'])
def delete_item(id):
    return jsonify({"deleted": id}), 204
```

---

## Routing și URL Parameters

### 1. Parametri în URL

```python
# Parametru string
@app.route('/user/<username>')
def show_user(username):
    return f"User: {username}"

# Parametru integer
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f"Post ID: {post_id}"

# Parametru float
@app.route('/price/<float:amount>')
def show_price(amount):
    return f"Price: {amount}"

# Parametru path (include slash-uri)
@app.route('/path/<path:subpath>')
def show_path(subpath):
    return f"Path: {subpath}"
```

### 2. Query Parameters

```python
from flask import request

@app.route('/search')
def search():
    # GET /search?q=python&limit=10
    query = request.args.get('q', '')
    limit = request.args.get('limit', 10, type=int)
    return jsonify({
        "query": query,
        "limit": limit
    })
```

---

## Request și Response

### Request Data

```python
from flask import request

@app.route('/api/data', methods=['POST'])
def handle_data():
    # JSON data
    json_data = request.json

    # Form data
    form_data = request.form

    # Query parameters
    args = request.args

    # Headers
    auth_header = request.headers.get('Authorization')

    # Files
    file = request.files.get('file')

    return jsonify({
        "json": json_data,
        "form": dict(form_data),
        "query": dict(args)
    })
```

### Response Types

```python
from flask import jsonify, make_response

# 1. JSON Response (cel mai comun)
@app.route('/api/json')
def json_response():
    return jsonify({"key": "value"}), 200

# 2. Custom Response
@app.route('/api/custom')
def custom_response():
    response = make_response(jsonify({"data": "test"}))
    response.headers['X-Custom-Header'] = 'Value'
    response.status_code = 200
    return response

# 3. Text Response
@app.route('/api/text')
def text_response():
    return "Plain text response", 200

# 4. Response cu headers custom
@app.route('/api/headers')
def headers_response():
    return jsonify({"status": "ok"}), 200, {
        'Content-Type': 'application/json',
        'X-API-Version': '1.0'
    }
```

---

## CRUD Operations

Exemplu complet pentru o aplicație de gestiune task-uri:

```python
from flask import Flask, jsonify, request

app = Flask(__name__)

# Bază de date simulată
tasks = [
    {"id": 1, "title": "Învață Flask", "completed": False},
    {"id": 2, "title": "Creează API", "completed": False}
]

# CREATE - POST
@app.route('/api/tasks', methods=['POST'])
def create_task():
    if not request.json or 'title' not in request.json:
        return jsonify({"error": "Title este obligatoriu"}), 400

    new_task = {
        "id": tasks[-1]['id'] + 1 if tasks else 1,
        "title": request.json['title'],
        "completed": request.json.get('completed', False)
    }
    tasks.append(new_task)

    return jsonify({
        "task": new_task,
        "message": "Task creat cu succes"
    }), 201

# READ - GET (toate)
@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    # Filtrare opțională
    completed = request.args.get('completed', type=bool)

    if completed is not None:
        filtered_tasks = [t for t in tasks if t['completed'] == completed]
        return jsonify({"tasks": filtered_tasks})

    return jsonify({
        "tasks": tasks,
        "count": len(tasks)
    })

# READ - GET (unul singur)
@app.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task(task_id):
    task = next((t for t in tasks if t['id'] == task_id), None)

    if task is None:
        return jsonify({"error": "Task nu a fost găsit"}), 404

    return jsonify({"task": task})

# UPDATE - PUT
@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task = next((t for t in tasks if t['id'] == task_id), None)

    if task is None:
        return jsonify({"error": "Task nu a fost găsit"}), 404

    if not request.json:
        return jsonify({"error": "Date invalide"}), 400

    task['title'] = request.json.get('title', task['title'])
    task['completed'] = request.json.get('completed', task['completed'])

    return jsonify({
        "task": task,
        "message": "Task actualizat cu succes"
    })

# DELETE - DELETE
@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    task = next((t for t in tasks if t['id'] == task_id), None)

    if task is None:
        return jsonify({"error": "Task nu a fost găsit"}), 404

    tasks.remove(task)

    return jsonify({"message": "Task șters cu succes"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=5000)
```

---

## Error Handling

### 1. Error Handlers

```python
from flask import jsonify

# Handler pentru 404
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Resursa nu a fost găsită",
        "status": 404
    }), 404

# Handler pentru 500
@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Eroare internă a serverului",
        "status": 500
    }), 500

# Handler pentru erori custom
@app.errorhandler(Exception)
def handle_exception(error):
    return jsonify({
        "error": str(error),
        "type": type(error).__name__
    }), 500
```

### 2. Custom Exceptions

```python
class ValidationError(Exception):
    status_code = 400

    def __init__(self, message, status_code=None):
        super().__init__()
        self.message = message
        if status_code is not None:
            self.status_code = status_code

@app.errorhandler(ValidationError)
def handle_validation_error(error):
    return jsonify({
        "error": error.message,
        "status": error.status_code
    }), error.status_code

# Utilizare
@app.route('/api/validate')
def validate():
    if not some_condition:
        raise ValidationError("Datele sunt invalide")
    return jsonify({"status": "ok"})
```

---

## Best Practices

### 1. Organizarea Codului

```python
# config.py
class Config:
    DEBUG = False
    SECRET_KEY = 'your-secret-key'

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

# app.py
from flask import Flask
from config import DevelopmentConfig

app = Flask(__name__)
app.config.from_object(DevelopmentConfig)
```

### 2. Blueprints (pentru proiecte mai mari)

```python
# routes/tasks.py
from flask import Blueprint, jsonify

tasks_bp = Blueprint('tasks', __name__)

@tasks_bp.route('/api/tasks')
def get_tasks():
    return jsonify({"tasks": []})

# app.py
from flask import Flask
from routes.tasks import tasks_bp

app = Flask(__name__)
app.register_blueprint(tasks_bp)
```

### 3. CORS (Cross-Origin Resource Sharing)

```python
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Permite toate originile

# Sau configurare detaliată
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000"],
        "methods": ["GET", "POST", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})
```

### 4. Environment Variables

```python
# .env
FLASK_APP=app.py
FLASK_ENV=development
SECRET_KEY=your-secret-key
DATABASE_URL=sqlite:///app.db

# app.py
from dotenv import load_dotenv
import os

load_dotenv()

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
```

### 5. Logging

```python
import logging

# Configurare logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

@app.route('/api/log-example')
def log_example():
    app.logger.info('Acest endpoint a fost accesat')
    app.logger.warning('Avertisment de test')
    app.logger.error('Eroare de test')
    return jsonify({"status": "check logs"})
```

---

## Testare API cu curl

```bash
# GET
curl http://localhost:5000/api/tasks

# POST
curl -X POST http://localhost:5000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "Task nou", "completed": false}'

# PUT
curl -X PUT http://localhost:5000/api/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"title": "Task actualizat", "completed": true}'

# DELETE
curl -X DELETE http://localhost:5000/api/tasks/1
```

---

## Testare API cu Python

```python
import requests

# GET
response = requests.get('http://localhost:5000/api/tasks')
print(response.json())

# POST
response = requests.post(
    'http://localhost:5000/api/tasks',
    json={"title": "Task nou", "completed": False}
)
print(response.json())

# PUT
response = requests.put(
    'http://localhost:5000/api/tasks/1',
    json={"title": "Task actualizat", "completed": True}
)
print(response.json())

# DELETE
response = requests.delete('http://localhost:5000/api/tasks/1')
print(response.json())
```

---

## Extensii Utile Flask

- **Flask-SQLAlchemy** - ORM pentru baze de date
- **Flask-Migrate** - Migrări baze de date
- **Flask-JWT-Extended** - Autentificare JWT
- **Flask-CORS** - Cross-Origin Resource Sharing
- **Flask-RESTful** - Building REST APIs
- **Flask-Limiter** - Rate limiting
- **Flask-Caching** - Caching

---

## Resurse Suplimentare

- [Documentație Oficială Flask](https://flask.palletsprojects.com/)
- [Flask Mega Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)
- [REST API Design Best Practices](https://restfulapi.net/)

---

## Concluzie

Flask este perfect pentru:
- API-uri REST simple și rapide
- Microservicii
- Prototipare rapidă
- Proiecte mici și medii

Puncte cheie:
- Simplu și ușor de învățat
- Foarte flexibil
- Comunitate mare și active
- Multe extensii disponibile