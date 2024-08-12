from flask import Flask, request, jsonify
from langchain_diagnosis import generate_diagnosis, handle_follow_up_questions



app = Flask(__name__)

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    print("Diagnosis endpoint hit")
    data = request.json
    result = str(generate_diagnosis(data['text']))
    return jsonify({"result": result})

@app.route('/follow_up', methods=['POST'])
def follow_up():
    print("follow up endpoint hit")
    data = request.json
    
    patient_diagnosis = data.get('patient_diagnosis')
    patient_query = data.get('patient_query')
    conversation_history = data.get('conversation_history')
    
   
    response, follow_up_suggestion = handle_follow_up_questions(patient_diagnosis, patient_query, conversation_history)
    return jsonify({
            "response": response,
            "follow_up_suggestion": follow_up_suggestion
        })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)