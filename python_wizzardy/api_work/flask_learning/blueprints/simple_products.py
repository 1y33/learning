from flask import Blueprint, jsonify, request

products = Blueprint('products', __name__)

product_list = [
    {"id": 1, "name": "Laptop", "price": 3500},
    {"id": 2, "name": "Mouse", "price": 50}
]

@products.route('/products')
def get_products():
    return jsonify(product_list)

@products.route('/products/<int:id>')
def get_product(id):
    product = next((p for p in product_list if p['id'] == id), None)
    if not product:
        return jsonify({"error": "Product not found"}), 404
    return jsonify(product)

@products.route('/products', methods=['POST'])
def add_product():
    new_product = {
        "id": len(product_list) + 1,
        "name": request.json['name'],
        "price": request.json['price']
    }
    product_list.append(new_product)
    return jsonify(new_product), 201
