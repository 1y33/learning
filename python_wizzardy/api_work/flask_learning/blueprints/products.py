from flask import Blueprint, jsonify, request

# Creează blueprint pentru produse
products_bp = Blueprint('products', __name__, url_prefix='/api/products')

# Date simulate
products = [
    {"id": 1, "name": "Laptop", "price": 3500, "stock": 10, "category": "Electronics"},
    {"id": 2, "name": "Mouse", "price": 50, "stock": 100, "category": "Accessories"},
    {"id": 3, "name": "Keyboard", "price": 150, "stock": 50, "category": "Accessories"}
]

# GET - Obține toate produsele (cu filtrare opțională)
@products_bp.route('/', methods=['GET'])
def get_products():
    # Filtrare după categorie (query parameter)
    category = request.args.get('category')

    if category:
        filtered = [p for p in products if p['category'].lower() == category.lower()]
        return jsonify({
            "products": filtered,
            "count": len(filtered),
            "filter": category
        }), 200

    return jsonify({
        "products": products,
        "count": len(products)
    }), 200

# GET - Obține un produs după ID
@products_bp.route('/<int:product_id>', methods=['GET'])
def get_product(product_id):
    product = next((p for p in products if p['id'] == product_id), None)

    if product is None:
        return jsonify({"error": "Produsul nu a fost găsit"}), 404

    return jsonify({"product": product}), 200

# POST - Creează un produs nou
@products_bp.route('/', methods=['POST'])
def create_product():
    required_fields = ['name', 'price', 'stock']

    if not request.json or not all(field in request.json for field in required_fields):
        return jsonify({"error": "Name, price și stock sunt obligatorii"}), 400

    new_product = {
        "id": products[-1]['id'] + 1 if products else 1,
        "name": request.json['name'],
        "price": request.json['price'],
        "stock": request.json['stock'],
        "category": request.json.get('category', 'General')
    }

    products.append(new_product)

    return jsonify({
        "product": new_product,
        "message": "Produs creat cu succes"
    }), 201

# PUT - Actualizează un produs
@products_bp.route('/<int:product_id>', methods=['PUT'])
def update_product(product_id):
    product = next((p for p in products if p['id'] == product_id), None)

    if product is None:
        return jsonify({"error": "Produsul nu a fost găsit"}), 404

    product['name'] = request.json.get('name', product['name'])
    product['price'] = request.json.get('price', product['price'])
    product['stock'] = request.json.get('stock', product['stock'])
    product['category'] = request.json.get('category', product['category'])

    return jsonify({
        "product": product,
        "message": "Produs actualizat cu succes"
    }), 200

# DELETE - Șterge un produs
@products_bp.route('/<int:product_id>', methods=['DELETE'])
def delete_product(product_id):
    product = next((p for p in products if p['id'] == product_id), None)

    if product is None:
        return jsonify({"error": "Produsul nu a fost găsit"}), 404

    products.remove(product)

    return jsonify({"message": "Produs șters cu succes"}), 200

# GET - Categorii disponibile
@products_bp.route('/categories', methods=['GET'])
def get_categories():
    categories = list(set(p['category'] for p in products))

    return jsonify({
        "categories": categories,
        "count": len(categories)
    }), 200
