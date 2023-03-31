from clams.src.clams.CLAMS import ClusterAmbiguity
## import flask
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

app = Flask(__name__)
CORS(app)

## test code hello world
@app.route('/', methods=['GET'])
def hello():
  return jsonify({
		"message": "Hello World"
	})

@app.route('/clams', methods=['POST'])
def clams():
 
 
  data = np.array(request.get_json()["data"])
  ca_obj = ClusterAmbiguity(S=1.0, mode="entropy")
  ambiguiy_score = ca_obj.fit(data)
  key_list = ca_obj.pair_key_list
  separability_list = 1 - ca_obj.filtered_prob_single_list
  ambiguity_list = ca_obj.entropy_list
  convariances = ca_obj.convariances
  means = ca_obj.means
  proba = ca_obj.proba
  
  return jsonify({
    "ambiguity_score": ambiguiy_score,
    "key_list": key_list,
    "separability_list": separability_list.tolist(),
    "ambiguity_list": ambiguity_list.tolist(),
    "covariances": convariances.tolist(),
    "means": means.tolist(),
    "proba": proba.tolist()
	})

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=9999, debug=True)