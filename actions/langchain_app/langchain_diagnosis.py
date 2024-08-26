# from langchain_core.prompts import PromptTemplate
from langchain_cohere.llms import Cohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from langchain_cohere import CohereEmbeddings
from langchain_core.messages import HumanMessage
from langchain_mistralai.chat_models import ChatMistralAI

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import pinecone
import logging

logging.basicConfig(level=logging.WARNING)

from flask import Flask, request, jsonify

import os
from dotenv import load_dotenv
import json

load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY") 
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# os.environ["COHERE_API_KEY"] = getpass.getpass()

def generate_diagnosis(patient_details):
    # pinecone_prompt_template = PromptTemplate(template="{query}")

    formatted_prompt = patient_details
    print("Formatted Prompt:", formatted_prompt)

    # embeddings = CohereEmbeddings(model="embed-english-light-v3.0", )  
    embeddings_model = CohereEmbeddings(cohere_api_key=COHERE_API_KEY, model="embed-english-light-v3.0")
    embedded_query = embeddings_model.embed_query(patient_details)

    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "medical-information"
    # index_metadata = pc.describe_index(index_name)

    # print(index_metadata['dimension'])

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(       
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 

    index = pc.Index(index_name)

    query_response_condition = index.query(
        vector=embedded_query,   
        top_k=3, 
        namespace='medical-conditions',                 
        include_values=True)
    
    query_response_medicine = index.query(
        vector=embedded_query,   
        top_k=3, 
        namespace='medicine-names',                 
        include_values=True)
    

    with open('wikipedia_medical_data.json') as f:
        medical_data = json.load(f)
    
    for match in query_response_condition['matches']:
        # print(f"ID: {match['id']}, Score: {match['score']}, Values: {match['values']}")
        pass

    for match in query_response_medicine['matches']:
        # print(f"ID: {match['id']}, Score: {match['score']}, Values: {match['values']}")
        pass

    matched_conditions = [medical_data[match['id']] for match in query_response_condition['matches']]
    # print(matched_conditions)

    matched_medicines = [medical_data[match['id']] for match in query_response_medicine['matches']]
    # print(matched_medicines)

    # embeddings_array = np.array(embeddings_list)

    # pca = PCA(n_components=512)
    # reduced_embeddings = pca.fit_transform(embeddings_array)
    # reduced_embeddings_list = reduced_embeddings.tolist()

    

    query_text = (
        f"You will respond as a compassionate and empathetic medical doctor with extensive knowledge of rare medical conditions."
        "Your patient, who has no access to medical care, has come to you with the report of a concerning medical problem they have been experiencing and is seeking your help."
        "Your task is to gently and carefully diagnose their condition based on the description of their symptoms and medical history they provide, and provide them with the the necessary, most compassionate medical advice on how to manage their condition."
        "Given the challenges they face in accessing medical care, you must provide a thorough diagnosis and considerate suggestions for treatment."
        "It’s crucial that you address the patient directly. Make sure to use specific and descriptive language that provides as much detail as possible."
        "Speak with kindness, understanding, and reassurance, acknowledging the patient's concerns and fears."
         "Remember, the patient is reading this, so address them. Also make sure to let them know that you are not a real doctor."
        "Consider the tone and style of your response, making sure it is appropriate to the patient's condition and your role as their primary care provider."
        "Use your extensive knowledge of rare medical conditions to provide the patient with the best possible medical advice and treatment along with also emotional support."
        f"Report: {patient_details}\n"
        "Provide a thoughtful diagnosis along with suggested over-the-counter medications, keeping in mind the patient's well-being."
        "This response will go to a doctor for verification before being sent to the end user.")


    formatted_prompt = f"Query: {query_text}\n Possible medical conditions that are most closely related to the patient symptoms and need to be accounted for:\n"
    for i, embedding in enumerate(matched_conditions):
        formatted_prompt += f"Embedding {i + 1}: {embedding}\n"

    formatted_prompt += "Possible medicine details that are most closely related to the patient symptoms : "
    for i, embedding in enumerate(matched_medicines):
        formatted_prompt += f"Embedding {i + 1}: {embedding}\n"

    model = Cohere(cohere_api_key=COHERE_API_KEY)

    chat = ChatMistralAI(model="mistral-small", api_key=MISTRAL_API_KEY)
    messages = [HumanMessage(content=formatted_prompt)]

    # print(model.invoke(messages))

    return model.invoke(formatted_prompt)



def handle_follow_up_questions(patient_diagnosis, patient_query, conversation_history):
    llm = ChatMistralAI(model="mistral-small", api_key=MISTRAL_API_KEY)


    prompt_template = ChatPromptTemplate.from_template("""
    Based on the following diagnosis and conversation history:

    Diagnosis: {diagnosis}
    Conversation History: {history}

    Answer the user's follow-up query:
    {query}
    """)

    # Create a sequence of runnables
    chain = prompt_template | llm | StrOutputParser()

    diagnosis =  patient_diagnosis
    history = conversation_history
    user_query = patient_query

    # Run the sequence to get a response
    response = chain.invoke({
        "diagnosis": diagnosis,
        "history": history,
        "query": user_query
    })

    print(response)
    print('\n\n\n\n\n\n\n\n')

    analysis_prompt = ChatPromptTemplate.from_template("What could be a potential/ helpful follow up question that a patient potentially diagnosed with this disease may ask? Just mention the question, no other information is required  {response}")

    composed_chain = (
        {"response": chain}
        | analysis_prompt
        | llm
        | StrOutputParser()
    )

    # Run the composed chain
    follow_up_suggestion = composed_chain.invoke({
        "diagnosis": diagnosis,
        "history": history,
        "query": user_query
    })

    print(follow_up_suggestion)

    return response, follow_up_suggestion



# app = Flask(__name__)

# @app.route('/diagnosis', methods=['POST'])
# def diagnosis():
#     print("Diagnosis endpoint hit")
#     data = request.json
#     result = generate_diagnosis(data['text'])
#     return jsonify(result)

# @app.route('/follow_up', methods=['POST'])
# def follow_up():
#     print("follow up endpoint hit")
#     data = request.json
    
#     patient_diagnosis = data.get('patient_diagnosis')
#     patient_query = data.get('patient_query')
#     conversation_history = data.get('conversation_history')
    
   
#     result = handle_follow_up_questions(patient_diagnosis, patient_query, conversation_history)
#     return jsonify(result)

# if __name__ == "__main__":
#     app.run(host='0.0.0.0', port=5001)
