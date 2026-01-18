from flask import Flask, jsonify
from blueprints.users import users_bp
from blueprints.products import products_bp
from blueprints.orders import orders_bp

# Creează aplicația Flask
app = Flask(__name__)

# Înregistrează blueprints-urile
app.register_blueprint(users_bp)
app.register_blueprint(products_bp)
app.register_blueprint(orders_bp)

# Route de bază
@app.route('/')
def home():
    return jsonify({
        "message": "API cu Blueprints",
        "version": "2.0",
        "endpoints": {
            "users": "/api/users",
            "products": "/api/products",
            "orders": "/api/orders"
        }
    })

# Health check
@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "service": "Flask API with Blueprints"
    }), 200

# Documentație API
@app.route('/api/docs')
def api_docs():
    return jsonify({
        "users_endpoints": {
            "GET /api/users": "Obține toți utilizatorii",
            "GET /api/users/<id>": "Obține un utilizator",
            "POST /api/users": "Creează un utilizator",
            "PUT /api/users/<id>": "Actualizează un utilizator",
            "DELETE /api/users/<id>": "Șterge un utilizator"
        },
        "products_endpoints": {
            "GET /api/products": "Obține toate produsele (filtrare: ?category=name)",
            "GET /api/products/<id>": "Obține un produs",
            "POST /api/products": "Creează un produs",
            "PUT /api/products/<id>": "Actualizează un produs",
            "DELETE /api/products/<id>": "Șterge un produs",
            "GET /api/products/categories": "Obține categoriile disponibile"
        },
        "orders_endpoints": {
            "GET /api/orders": "Obține toate comenzile (filtrare: ?status=pending, ?user_id=1)",
            "GET /api/orders/<id>": "Obține o comandă",
            "POST /api/orders": "Creează o comandă",
            "PUT /api/orders/<id>": "Actualizează status comandă",
            "DELETE /api/orders/<id>": "Anulează o comandă",
            "GET /api/orders/stats": "Statistici comenzi"
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        "error": "Endpoint-ul nu a fost găsit",
        "status": 404
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        "error": "Eroare internă a serverului",
        "status": 500
    }), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 API Flask cu Blueprints pornit!")
    print("="*50)
    print("\n📍 Endpoints disponibile:")
    print("   - Home: http://127.0.0.1:5000/")
    print("   - Docs: http://127.0.0.1:5000/api/docs")
    print("   - Users: http://127.0.0.1:5000/api/users")
    print("   - Products: http://127.0.0.1:5000/api/products")
    print("   - Orders: http://127.0.0.1:5000/api/orders")
    print("\n" + "="*50 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
