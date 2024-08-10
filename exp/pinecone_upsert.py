import pinecone
import pickle
import numpy as np
from unidecode import unidecode

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

import os
from dotenv import load_dotenv

load_dotenv()


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

pc = Pinecone(
    api_key=PINECONE_API_KEY
)

index_name = "medical-information"
# index_metadata = pc.describe_index(index_name)

# print(index_metadata['dimension'])
with open('medical_embeddings_light.pkl', 'rb') as f:
    loaded_embeddings = pickle.load(f)

loaded_embeddings_unidecode = {unidecode(med): emb for med, emb in loaded_embeddings.items()}

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

# vectors = [(med, emb) for med, emb in medical_embeddings.items()]
# index.upsert(vectors)

# print('embeddings stored')

embeddings_array = np.array(loaded_embeddings_unidecode)


def batch_upsert(index, vectors, batch_size=100):
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i+batch_size]
        index.upsert(batch)


vectors = [(med, emb) for med, emb in loaded_embeddings_unidecode.items()]
batch_upsert(index, vectors)

print("Embeddings upserted successfully.")