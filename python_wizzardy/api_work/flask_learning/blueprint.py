from flask import Blueprint,jsonify


job = Blueprint("job",__name__)

@job.route("/")
def home():
    return jsonify(
        "test"
    )
    
    
work = Blueprint("work",__name__)
@work = Blueprint("/")
def home():
    return jsonify("
                   test")