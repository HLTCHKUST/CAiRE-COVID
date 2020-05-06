from flask import Flask, request, jsonify
from lucene_update import get_results, get_para_results
app = Flask(__name__)

@app.route('/query', methods=['POST'])
def return_matches():
    content = request.json
    out = get_results(content['text'])
    return jsonify(out)

@app.route('/query_paragraph', methods=['POST'])
def return_para_matches():
    content = request.json
    out = get_para_results(content['text'])
    return jsonify(out)

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)
