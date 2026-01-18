from flask import Blueprint, jsonify, request
from datetime import datetime

# Creează blueprint pentru comenzi
orders_bp = Blueprint('orders', __name__, url_prefix='/api/orders')

# Date simulate
orders = [
    {
        "id": 1,
        "user_id": 1,
        "products": [
            {"product_id": 1, "quantity": 2, "price": 3500}
        ],
        "total": 7000,
        "status": "pending",
        "created_at": "2024-01-15"
    }
]

# GET - Obține toate comenzile
@orders_bp.route('/', methods=['GET'])
def get_orders():
    # Filtrare după status
    status = request.args.get('status')
    user_id = request.args.get('user_id', type=int)

    filtered_orders = orders

    if status:
        filtered_orders = [o for o in filtered_orders if o['status'].lower() == status.lower()]

    if user_id:
        filtered_orders = [o for o in filtered_orders if o['user_id'] == user_id]

    return jsonify({
        "orders": filtered_orders,
        "count": len(filtered_orders)
    }), 200

# GET - Obține o comandă după ID
@orders_bp.route('/<int:order_id>', methods=['GET'])
def get_order(order_id):
    order = next((o for o in orders if o['id'] == order_id), None)

    if order is None:
        return jsonify({"error": "Comanda nu a fost găsită"}), 404

    return jsonify({"order": order}), 200

# POST - Creează o comandă nouă
@orders_bp.route('/', methods=['POST'])
def create_order():
    if not request.json or 'user_id' not in request.json or 'products' not in request.json:
        return jsonify({"error": "user_id și products sunt obligatorii"}), 400

    # Calculează totalul
    total = sum(item['price'] * item['quantity'] for item in request.json['products'])

    new_order = {
        "id": orders[-1]['id'] + 1 if orders else 1,
        "user_id": request.json['user_id'],
        "products": request.json['products'],
        "total": total,
        "status": request.json.get('status', 'pending'),
        "created_at": datetime.now().strftime('%Y-%m-%d')
    }

    orders.append(new_order)

    return jsonify({
        "order": new_order,
        "message": "Comandă creată cu succes"
    }), 201

# PUT - Actualizează status comandă
@orders_bp.route('/<int:order_id>', methods=['PUT'])
def update_order(order_id):
    order = next((o for o in orders if o['id'] == order_id), None)

    if order is None:
        return jsonify({"error": "Comanda nu a fost găsită"}), 404

    # Actualizează doar status-ul (comenzile nu ar trebui modificate complet)
    order['status'] = request.json.get('status', order['status'])

    return jsonify({
        "order": order,
        "message": f"Comandă actualizată la status: {order['status']}"
    }), 200

# DELETE - Anulează o comandă
@orders_bp.route('/<int:order_id>', methods=['DELETE'])
def cancel_order(order_id):
    order = next((o for o in orders if o['id'] == order_id), None)

    if order is None:
        return jsonify({"error": "Comanda nu a fost găsită"}), 404

    if order['status'] == 'completed':
        return jsonify({"error": "Nu se poate anula o comandă completată"}), 400

    order['status'] = 'cancelled'

    return jsonify({
        "order": order,
        "message": "Comandă anulată cu succes"
    }), 200

# GET - Statistici comenzi
@orders_bp.route('/stats', methods=['GET'])
def get_stats():
    total_orders = len(orders)
    total_revenue = sum(o['total'] for o in orders if o['status'] != 'cancelled')

    status_counts = {}
    for order in orders:
        status = order['status']
        status_counts[status] = status_counts.get(status, 0) + 1

    return jsonify({
        "total_orders": total_orders,
        "total_revenue": total_revenue,
        "status_breakdown": status_counts
    }), 200
