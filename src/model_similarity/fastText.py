from gensim.models.fasttext import FastText as FT
from scipy.spatial.distance import cosine
import torch
import numpy as np
import pandas as pd
import os

# # Calculate FastText embedding for a single text
def get_fast_text_embedding(text,model,embedding_size=300):
    words = text.split()
    
    word_embeddings = []
    for word in words:
        try:
            word_embeddings.append(model.wv[word])  
        except KeyError:
            continue 
            
    if len(word_embeddings) == 0:
        return np.zeros(embedding_size)  
    
    text_embedding = np.mean(word_embeddings, axis=0)
    return text_embedding


# Function to calculate similarity using SentenceTransformer
def calculate_fasttext_similarity(texts1, texts2, model,batch_size=8):


    # NaN or None
    texts1 = ["" if pd.isna(text) else text for text in texts1]
    texts2 = ["" if pd.isna(text) else text for text in texts2]

    # Initialize lists to store embeddings
    embeddings1 = []
    embeddings2 = []
    
    
    # Process in batches to save memory
    for i in range(0, len(texts1), batch_size):
        batch1 = texts1[i:i+batch_size]
        batch2 = texts2[i:i+batch_size]
            
        # Get embeddings for the current batch
        batch_embeddings1 = [get_fast_text_embedding(text,model) for text in batch1]
        batch_embeddings2 = [get_fast_text_embedding(text,model) for text in batch2]

        embeddings1.extend(batch_embeddings1)
        embeddings2.extend(batch_embeddings2)
    
    # Calculate cosine similarity between the embeddings
    similarities = []
    for v1, v2 in zip(embeddings1, embeddings2):
        if np.all(v1 == 0) or np.all(v2 == 0):  #zero vectors
            similarities.append(0)  
        else:
            similarities.append(1 - cosine(v1, v2))
    return similarities


