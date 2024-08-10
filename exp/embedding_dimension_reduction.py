import pickle
import numpy as np
from unidecode import unidecode

with open('medical_embeddings.pkl', 'rb') as f:
    loaded_embeddings = pickle.load(f)

loaded_embeddings_unidecode = {unidecode(med): emb for med, emb in loaded_embeddings.items()}

import numpy as np

# Convert the embeddings to a NumPy array
embeddings_list = list(loaded_embeddings_unidecode.values())
embeddings_array = np.array(embeddings_list)

from sklearn.decomposition import PCA

# Initialize PCA to reduce dimensions from 1024 to 100
pca = PCA(n_components=100)
reduced_embeddings_array = pca.fit_transform(embeddings_array)

# Convert back to dictionary
reduced_embeddings_unidecode = {
    med: reduced_emb for med, reduced_emb in zip(loaded_embeddings_unidecode.keys(), reduced_embeddings_array)
}


# Save reduced embeddings to a file
with open('reduced_medical_embeddings.pkl', 'wb') as f:
    pickle.dump(reduced_embeddings_unidecode, f)

