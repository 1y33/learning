from flask import Blueprint, jsonify, request

# Creează blueprint pentru users
users_bp = Blueprint('users', __name__, url_prefix='/api/users')

# Date simulate (în loc de bază de date)
users = [
    {"id": 1, "name": "Ion Popescu", "email": "ion@example.com", "role": "admin"},
    {"id": 2, "name": "Maria Ionescu", "email": "maria@example.com", "role": "user"}
]

# GET - Obține toți utilizatorii
@users_bp.route('/', methods=['GET'])
def get_users():
    return jsonify({
        "users": users,
        "count": len(users)
    }), 200

# GET - Obține un utilizator după ID
@users_bp.route('/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)

    if user is None:
        return jsonify({"error": "Utilizatorul nu a fost găsit"}), 404

    return jsonify({"user": user}), 200

# POST - Creează un utilizator nou
@users_bp.route('/', methods=['POST'])
def create_user():
    if not request.json or 'name' not in request.json or 'email' not in request.json:
        return jsonify({"error": "Name și email sunt obligatorii"}), 400

    new_user = {
        "id": users[-1]['id'] + 1 if users else 1,
        "name": request.json['name'],
        "email": request.json['email'],
        "role": request.json.get('role', 'user')
    }

    users.append(new_user)

    return jsonify({
        "user": new_user,
        "message": "Utilizator creat cu succes"
    }), 201

# PUT - Actualizează un utilizator
@users_bp.route('/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)

    if user is None:
        return jsonify({"error": "Utilizatorul nu a fost găsit"}), 404

    user['name'] = request.json.get('name', user['name'])
    user['email'] = request.json.get('email', user['email'])
    user['role'] = request.json.get('role', user['role'])

    return jsonify({
        "user": user,
        "message": "Utilizator actualizat cu succes"
    }), 200

# DELETE - Șterge un utilizator
@users_bp.route('/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    user = next((u for u in users if u['id'] == user_id), None)

    if user is None:
        return jsonify({"error": "Utilizatorul nu a fost găsit"}), 404

    users.remove(user)

    return jsonify({"message": "Utilizator șters cu succes"}), 200
