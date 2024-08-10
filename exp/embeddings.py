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


with open('wikipedia_medical_data.json') as f:
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

with open('medical_embeddings_light.pkl', 'wb') as f:
    pickle.dump(medical_embeddings, f)

print('saved embeddings')

# pinecone.init(api_key='your-pinecone-api-key', environment='us-west1-gcp')

# # # Create or connect to a Pinecone index
# # index_name = 'medications'
# # if index_name not in pinecone.list_indexes():
# #     pinecone.create_index(index_name, dimension=len(embeddings[0]))

# # index = pinecone.Index(index_name)

# # # Upsert embeddings into the index
# # vectors = [(med, emb) for med, emb in medication_embeddings.items()]
# # index.upsert(vectors)

# pc = Pinecone(
#     api_key=PINECONE_API_KEY
# )

# index_name = "medical-information"
# # index_metadata = pc.describe_index(index_name)

# # print(index_metadata['dimension'])

# if index_name not in pc.list_indexes().names():
#     pc.create_index(
#         name=index_name,
#         dimension=len(embeddings[0]),
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud='aws', 
#             region='us-east-1'
#         ) 
#     ) 

# index = pc.Index(index_name)

# # vectors = [(med, emb) for med, emb in medical_embeddings.items()]
# # index.upsert(vectors)

# # print('embeddings stored')

# def batch_upsert(index, vectors, batch_size=100):
#     for i in range(0, len(vectors), batch_size):
#         batch = vectors[i:i+batch_size]
#         index.upsert(batch)

# # Step 10: Upsert embeddings into the index in batches
# vectors = [(med, emb) for med, emb in medical_embeddings.items()]
# batch_upsert(index, vectors)

# print("Embeddings upserted successfully.")