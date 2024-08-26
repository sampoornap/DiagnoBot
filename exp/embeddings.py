import json
import cohere
import pinecone
import pickle

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import os
from dotenv import load_dotenv

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")


with open('wikipedia_medicine_name_data.json') as f:
    medical_data = json.load(f)

co = cohere.Client(api_key=COHERE_API_KEY)

texts = list(medical_data.values())

def get_embeddings(texts):
    response = co.embed(texts=texts,
                model='embed-english-light-v3.0', 
                input_type='search_document')
    
    return response.embeddings

embeddings = get_embeddings(texts)
# print(len(embeddings))
# print(len(embeddings[0]))
medical_embeddings = {med: emb for med, emb in zip(medical_data.keys(), embeddings)}

print('done with embeddings')

with open('medicine_name_embeddings_light.pkl', 'wb') as f:
    pickle.dump(medical_embeddings, f)

print('saved embeddings')
