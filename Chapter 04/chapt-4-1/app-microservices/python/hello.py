
from flask import Flask
from flask import jsonify

app = Flask(__name__)

@app.route('/')
def hello():
    dictionary = {'Welcome to Developing Cloud Native Apps on GCP': 'You have successfully deployed a python microservice in GAE'}
    return jsonify(dictionary)

if __name__ == '__main__':
    app.run(debug=True)
