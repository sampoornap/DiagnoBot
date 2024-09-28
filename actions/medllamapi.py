# -*- coding: utf-8 -*-
"""medllamapi.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1gi4dl9ijNEotEDmfbJwZhBt__TlzuZV-
"""
"""
!pip install transformers   bitsandbytes  accelerate
!pip install flask
!pip install pyngrok
"""
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

model_name = "ruslanmv/Medical-Llama3-8B"
device_map = 'auto'
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    trust_remote_code=True,
    use_cache=False,
    device_map=device_map
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

sys_message = '''
    You are an AI Medical Assistant. Your role is to assist with medical queries based on symptoms and patient details provided. Use your medical knowledge to provide accurate, specific advice, treatment plans and over-the-counter medicine suggestions. Ask clarifying questions (such as food habits, sleep schedule, any ongoing medication, etc) if needed.
    '''

conversation_history = sys_message

def askme(conversation_history):
    sys_message = '''
    You are an AI Medical Assistant. Your role is to assist with medical queries based on symptoms and patient details provided. Use your medical knowledge to provide accurate, specific advice, treatment plans and over-the-counter medicine suggestions. You may ask clarifying questions (such as food habits, sleep schedule, any ongoing medication, etc) if needed for diagnosis.
    '''

    # prompt = f"<|im_start|>system\n{sys_message}\n<|im_end|>\n<|im_start|>user\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
    prompt = conversation_history + "Assistant:"

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=800, use_cache=True)

    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def extract_assistant_message(response):

      if "Assistant:" in response:

          assistant_part = response.split("Assistant:")[1]

          if "User:" in assistant_part:
              assistant_message = assistant_part.split("User:")[0]
          else:
              assistant_message = assistant_part  # No 'User:' means take everything after 'Assistant:'

          return assistant_message.strip()

      return ""

    response = extract_assistant_message(response_text)
    print(response)
    return response

print(askme('''I have fever and frequent headache throughout the day. Age: 34 year old woman. Past conditions: I have had a viral infection 2 months back. Additional Info: Nothing for now.
What steps should i take to get a proper diagnosis and what are the treatment options. Also let me know in case any other detail needs to be provided for diagnosis'''))

def trim_response(assistant_response):
    last_period_index = assistant_response.rfind('.')
    last_question_index = assistant_response.rfind('?')

    if last_period_index == -1 and last_question_index == -1:
        # If neither is found, return the response as is
        trimmed_response = assistant_response
    elif last_period_index == -1:
        # If only a question mark is found
        trimmed_response = assistant_response[:last_question_index + 1]
    elif last_question_index == -1:
        # If only a period is found
        trimmed_response = assistant_response[:last_period_index + 1]
    else:
        # If both are found, use the one that appears last
        closest_index = max(last_period_index, last_question_index)
        trimmed_response = assistant_response[:closest_index + 1]

    return trimmed_response

sys_message = '''
    You are an AI Medical Assistant. Your role is to assist with medical queries based on symptoms and patient details provided. Use your medical knowledge to provide accurate, specific advice, treatment plans and over-the-counter medicine suggestions.
    '''

"""!ngrok config add-authtoken '2m8rlzO9kKiuriqkVb8BbR1p9IJ_2rnGjSWwcK55g1GsUEnCD'"""

from flask import Flask, request, jsonify
from pyngrok import ngrok

app = Flask(__name__)


public_url = ngrok.connect(5000)
print(f" * ngrok tunnel URL: {public_url}")

first_question = True
conversation_history = sys_message+"\n"
@app.route('/process_query', methods=['POST'])
def process_query():
    global first_question
    global conversation_history
    print(first_question)
    print(conversation_history)
    data = request.json
    question = data.get("question", "")

    if first_question:
      question = question + "What steps should i take to get a proper diagnosis and what are the treatment options"
      first_question = False


    print(f"Received question: {question}")

    conversation_history = conversation_history + f"User: {question}\n"

    response = trim_response(askme(conversation_history))
    print(f"Generated response: {response}")

    conversation_history = conversation_history + f"Some previous information: {response}\n"
    print("---------------------------------------------------------------------------")
    return jsonify({"response": response})

app.run(port=5000)

