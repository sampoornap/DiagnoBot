from langchain_core.prompts import PromptTemplate
from langchain_cohere.llms import Cohere

from langchain_cohere import CohereEmbeddings

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import pinecone

from sklearn.decomposition import PCA
import numpy as np

import json

import os
import getpass
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# os.environ["COHERE_API_KEY"] = getpass.getpass()

def generate_diagnosis(patient_details, open_book=False):
    pinecone_prompt_template = PromptTemplate(template="{query}")

    formatted_prompt = pinecone_prompt_template.format(query=patient_details)
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



generate_diagnosis("i have a really sore throat and back pain")











"""

embeddings_model = CohereEmbeddings( model="embed-english-light-v3.0")


# Define a function to retrieve medication info from Pinecone based on a query
def get_cohere_embeddings(text):
    embedded_query = embeddings_model.embed_query(text)
    return embedded_query

# Define a function to generate a diagnosis using Cohere's text generation
# def generate_text_with_cohere(prompt):
#     response = co.generate(
#         model='command-xlarge-nightly',
#         prompt=prompt,
#         max_tokens=300,
#         temperature=0.5
#     )
#     return response.generations[0].text
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)


index_name = "medical-information"
index = pc.Index(index_name)

# embedding_retriever = EmbeddingRetrievalChain(
#     embeddings=embeddings_model,
#     pinecone_index=index,
#     top_k=5,
#     include_metadata=True
# )
# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["patient_details", "medication_info"],
    template=\"""Patient Details: {patient_details}

Medications Info: {medication_info}

Based on the above information, provide a diagnosis and any recommended treatments.\"""
)
with open('wikipedia_medical_data.json') as f:
    medication_data = json.load(f)

# Initialize the LLM with Cohere
llm = Cohere(model='command-xlarge-nightly')

# Define a function to retrieve medication info from Pinecone based on a query
def retrieve_medication_info(query):
    query_embedding = get_cohere_embeddings(query)
    result = index.query([query_embedding], top_k=5, include_metadata=True)
    medications = [res['id'] for res in result['matches']]
    return "\n\n".join([medication_data[med] for med in medications])
prompt_template = PromptTemplate(
    input_variables=["patient_details", "medication_info"],
    template=\"""Patient Details: {patient_details}

    Medications Info: {medication_info}

    Based on the above information, provide a diagnosis and any recommended treatments.\"""
)

# Initialize the LLM with Cohere
llm = Cohere(api_key='your-cohere-api-key', model='command-xlarge-nightly')

# Define the LLMPrompt
# llm_prompt = LLMPrompt(
#     llm=llm,
#     prompt_template=prompt_template
# )
# Define the function to generate a diagnosis
def generate_diagnosis(patient_details, use_pinecone=False):
    if use_pinecone:
        # Retrieve medication info from Pinecone
        medication_info = retrieve_medication_info(patient_details)
    else:
        medication_info = ""

    # Combine the patient details and medication info into the prompt
    combined_prompt = prompt_template.render(patient_details=patient_details, medication_info=medication_info)

    # Generate the response using the LLM
    response = llm.generate(combined_prompt)
    return response['generations'][0]['text']

# Example usage
patient_details = "Patient has symptoms of bacterial pneumonia and acne."

# Scenario 1: Use Pinecone to get additional medication info
diagnosis_with_pinecone = generate_diagnosis(patient_details, use_pinecone=True)
print("Diagnosis with Pinecone info:\n", diagnosis_with_pinecone)

# Scenario 2: Do not use Pinecone, just generate based on user input
diagnosis_without_pinecone = generate_diagnosis(patient_details, use_pinecone=False)
print("Diagnosis without Pinecone info:\n", diagnosis_without_pinecone)


"""