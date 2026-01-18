from flask import Flask, jsonify
from blueprints.simple_users import users
from blueprints.simple_products import products

app = Flask(__name__)

app.register_blueprint(users)
app.register_blueprint(products)

@app.route('/')
def home():
    return jsonify({
        "message": "Simple API",
        "endpoints": ["/users", "/products"]
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
