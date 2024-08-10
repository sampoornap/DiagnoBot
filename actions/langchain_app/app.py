from flask import Flask, request, jsonify
import cohere
from langchain_diagnosis import generate_diagnosis

app = Flask(__name__)

# Initialize Cohere client and Langchain model

@app.route('/process', methods=['POST'])
def process():
    data = request.json
    # Use Langchain and Cohere to process the data
    result = generate_diagnosis(data['text'])
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)