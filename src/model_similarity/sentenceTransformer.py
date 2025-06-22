from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import numpy as np
import pandas as pd
import torch

def get_sentence_transformer_embedding(texts):
    embedding_size=384

    # NaN or None check
    texts = [" " if pd.isna(text) else text for text in texts]

    model = SentenceTransformer('all-MiniLM-L6-v2')  # output dim = 384

    embeddings = []

    for text in texts:
        if not text.strip():  # text is blank, empty string
            embeddings.append(np.zeros(embedding_size))
        else:
            emb = model.encode([text])[0]
            embeddings.append(emb)

    return embeddings