from flask import Flask, request, jsonify
from test_lucene import getResults
app = Flask(__name__)

@app.route('/query_paragraph', methods=['POST'])
def return_matches():
    content = request.json
    out = getResults(content['text'])
    return jsonify(out)

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=4000, debug=True)
