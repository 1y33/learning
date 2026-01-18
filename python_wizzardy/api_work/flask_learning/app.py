from flask import Flask,jsonify,request

app = Flask(__name__)

@app.route('/hello')
def hello():
    return "\nHello  World"

@app.route('/')
def home():
    return jsonify({
        "message": "API",
        "version" : "1.0"
    })
    
@app.route('/items',methods=["GET"])
def create_rout():
    return jsonify({
        "data": [],
        "message": "Lista de items"
    })

@app.route('/items',methods=['POST'])
def create_items():
    data = request.json
    return jsonify({
        "created" : data,
        "test" : " test"
    })
    
if __name__ == "__main__":
    app.run(debug=True)
    
    
    