# from langchain_core.prompts import PromptTemplate
from langchain_cohere.llms import Cohere

from langchain_cohere import CohereEmbeddings

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import pinecone


from flask import Flask, request, jsonify

import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# os.environ["COHERE_API_KEY"] = getpass.getpass()

def generate_diagnosis(patient_details, open_book=False):
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
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws', 
                region='us-east-1'
            ) 
        ) 

    index = pc.Index(index_name)

    query_response = index.query(
        vector=embedded_query,   
        top_k=1,                  
        include_values=True)
    
    for match in query_response['matches']:
        print(f"ID: {match['id']}, Score: {match['score']}, Values: {match['values']}")

    embeddings_list = [match['values'] for match in query_response['matches']]
    print(len(embeddings_list[0]))
    # embeddings_array = np.array(embeddings_list)

    # pca = PCA(n_components=512)
    # reduced_embeddings = pca.fit_transform(embeddings_array)
    # reduced_embeddings_list = reduced_embeddings.tolist()

    

    query_text = (
        f"You will respond as a medical doctor with extensive knowledge of rare medical conditions."
                      "Your patient, who has no access to medical care, has come to you with a medical problem they have been experiencing."
                       "Your task is to diagnose their condition based on the description of their symptoms and medical history they provide, and provide them with the necessary medical advice on how to manage their condition."
                        "Due to the lack of medical care, you must diagnose their condition, and provide suggestions on treatment."
                        "Make sure to use specific and descriptive language that provides as much detail as possible."
                        "Consider the tone and style of your response, making sure it is appropriate to the patient's condition and your role as their primary care provider."
                        "Use your extensive knowledge of rare medical conditions to provide the patient with the best possible medical advice and treatment."
                      f"Report: {patient_details}\n"
                      f"Provide the diagnosis along with suggested medications. This response will go to a doctor for verification before being sent to the end user. ")
        

    formatted_prompt = f"Query: {query_text}\nRelated Embeddings:\n"
    for i, embedding in enumerate(embeddings_list):
        formatted_prompt += f"Embedding {i + 1}: {embedding}\n"

    model = Cohere(cohere_api_key=COHERE_API_KEY)

    print(model.invoke(formatted_prompt))

    return model.invoke(formatted_prompt)


print(generate_diagnosis("i have a heartache and chest pain"))




app = Flask(__name__)

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    data = request.json
    # Perform diagnosis using langchain_cohere
    result = generate_diagnosis(data)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)