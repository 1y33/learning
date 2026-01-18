from flask import Blueprint, jsonify, request

users = Blueprint('users', __name__)

user_list = [
    {"id": 1, "name": "Ion"},
    {"id": 2, "name": "Maria"}
]

@users.route('/users')
def get_users():
    return jsonify(user_list)

@users.route('/users/<int:id>')
def get_user(id):
    user = next((u for u in user_list if u['id'] == id), None)
    if not user:
        return jsonify({"error": "User not found"}), 404
    return jsonify(user)

@users.route('/users', methods=['POST'])
def add_user():
    new_user = {
        "id": len(user_list) + 1,
        "name": request.json['name']
    }
    user_list.append(new_user)
    return jsonify(new_user), 201
