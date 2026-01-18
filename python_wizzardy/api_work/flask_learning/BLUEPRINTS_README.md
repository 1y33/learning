# Flask Blueprints - Ghid Complet

## Ce sunt Blueprints?

**Blueprints** sunt o modalitate de a **organiza** aplicații Flask mari în componente mai mici și modulare.

### Avantaje:
- ✅ **Organizare** - Cod mai curat și structurat
- ✅ **Reutilizare** - Blueprints-uri pot fi folosite în mai multe aplicații
- ✅ **Separare** - Fiecare blueprint gestionează propria logică
- ✅ **Scalabilitate** - Ușor de extins aplicația

---

## Structura Proiectului

```
flask_learning/
├── app_blueprints.py          # Aplicația principală
├── blueprints/
│   ├── __init__.py            # Face blueprints un pachet
│   ├── users.py               # Blueprint pentru users
│   ├── products.py            # Blueprint pentru products
│   └── orders.py              # Blueprint pentru orders
└── BLUEPRINTS_README.md       # Acest fișier
```

---

## Cum Funcționează?

### 1. Crearea unui Blueprint

```python
# blueprints/users.py
from flask import Blueprint, jsonify

# Sintaxă: Blueprint(name, import_name, url_prefix)
users_bp = Blueprint('users', __name__, url_prefix='/api/users')

@users_bp.route('/')
def get_users():
    return jsonify({"users": []})
```

### 2. Înregistrarea în app.py

```python
# app_blueprints.py
from flask import Flask
from blueprints.users import users_bp

app = Flask(__name__)
app.register_blueprint(users_bp)  # Înregistrează blueprint-ul
```

### 3. Rezultatul

- Blueprint: `@users_bp.route('/')`
- URL prefix: `/api/users`
- **URL final**: `/api/users/`

---

## Rulare și Testare

### 1. Pornește serverul:

```bash
cd /home/cata/workspace/dev/learning/python_wizzardy/api_work/flask_learning
python app_blueprints.py
```

### 2. Testează endpoints:

```bash
# Home
curl http://127.0.0.1:5000/

# Documentație
curl http://127.0.0.1:5000/api/docs

# Users
curl http://127.0.0.1:5000/api/users

# Products
curl http://127.0.0.1:5000/api/products

# Orders
curl http://127.0.0.1:5000/api/orders
```

---

## Exemple de Utilizare

### USERS Blueprint

```bash
# GET - Toți utilizatorii
curl http://127.0.0.1:5000/api/users

# GET - Un utilizator
curl http://127.0.0.1:5000/api/users/1

# POST - Creează utilizator
curl -X POST http://127.0.0.1:5000/api/users \
  -H "Content-Type: application/json" \
  -d '{"name": "Andrei Pop", "email": "andrei@example.com", "role": "user"}'

# PUT - Actualizează utilizator
curl -X PUT http://127.0.0.1:5000/api/users/1 \
  -H "Content-Type: application/json" \
  -d '{"name": "Ion Popescu Updated", "role": "admin"}'

# DELETE - Șterge utilizator
curl -X DELETE http://127.0.0.1:5000/api/users/2
```

### PRODUCTS Blueprint

```bash
# GET - Toate produsele
curl http://127.0.0.1:5000/api/products

# GET - Filtrare după categorie
curl "http://127.0.0.1:5000/api/products?category=Electronics"

# GET - Categorii disponibile
curl http://127.0.0.1:5000/api/products/categories

# POST - Creează produs
curl -X POST http://127.0.0.1:5000/api/products \
  -H "Content-Type: application/json" \
  -d '{"name": "Monitor", "price": 800, "stock": 15, "category": "Electronics"}'

# PUT - Actualizează produs
curl -X PUT http://127.0.0.1:5000/api/products/1 \
  -H "Content-Type: application/json" \
  -d '{"price": 3200, "stock": 8}'

# DELETE - Șterge produs
curl -X DELETE http://127.0.0.1:5000/api/products/2
```

### ORDERS Blueprint

```bash
# GET - Toate comenzile
curl http://127.0.0.1:5000/api/orders

# GET - Filtrare după status
curl "http://127.0.0.1:5000/api/orders?status=pending"

# GET - Filtrare după user
curl "http://127.0.0.1:5000/api/orders?user_id=1"

# GET - Statistici
curl http://127.0.0.1:5000/api/orders/stats

# POST - Creează comandă
curl -X POST http://127.0.0.1:5000/api/orders \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 1,
    "products": [
      {"product_id": 1, "quantity": 2, "price": 3500},
      {"product_id": 2, "quantity": 1, "price": 50}
    ]
  }'

# PUT - Actualizează status
curl -X PUT http://127.0.0.1:5000/api/orders/1 \
  -H "Content-Type: application/json" \
  -d '{"status": "completed"}'

# DELETE - Anulează comandă
curl -X DELETE http://127.0.0.1:5000/api/orders/1
```

---

## Parametri Blueprint

```python
Blueprint(
    name='users',              # Nume unic
    import_name=__name__,      # Modulul curent
    url_prefix='/api/users',   # Prefix pentru toate rutele
    template_folder='templates',  # Opțional: folder template-uri
    static_folder='static'     # Opțional: folder fișiere statice
)
```

---

## Diferența față de app.py simplu

### Fără Blueprints:
```python
# app.py - tot într-un singur fișier
@app.route('/api/users')
def get_users():
    pass

@app.route('/api/products')
def get_products():
    pass

@app.route('/api/orders')
def get_orders():
    pass
# ... multe alte rute
```

### Cu Blueprints:
```python
# app.py - foarte simplu
app.register_blueprint(users_bp)
app.register_blueprint(products_bp)
app.register_blueprint(orders_bp)

# blueprints/users.py - separat
@users_bp.route('/')
def get_users():
    pass

# blueprints/products.py - separat
@products_bp.route('/')
def get_products():
    pass
```

---

## Best Practices

1. **Un Blueprint per resursa** - users, products, orders
2. **URL prefix consistent** - `/api/users`, `/api/products`
3. **Nume descriptive** - `users_bp`, `products_bp`
4. **Fișiere separate** - fiecare blueprint în propriul fișier
5. **Organizare logică** - grupează funcționalități similare

---

## Când să folosești Blueprints?

✅ **DA** - când ai:
- Mai mult de 5-10 rute
- Resurse diferite (users, products, orders)
- Logică complexă
- Echipă mare (fiecare lucrează la un blueprint)

❌ **NU** - când ai:
- API simplu cu 2-3 rute
- Prototip rapid
- Aplicație foarte mică

---

## Extensii Utile

```python
# CORS pentru blueprints
from flask_cors import CORS

app = Flask(__name__)
app.register_blueprint(users_bp)
CORS(users_bp)  # Doar pentru users

# Sau pentru toate
CORS(app)
```

---

## Concluzie

Blueprints sunt **esențiale** pentru aplicații Flask mari și bine organizate. Ele fac codul:
- Mai ușor de întreținut
- Mai ușor de testat
- Mai ușor de extins
- Mai profesional

**Recomandare**: Folosește blueprints din start dacă știi că aplicația va crește!
