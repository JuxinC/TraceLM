import gensim.downloader as api
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from nltk.tokenize import word_tokenize

def get_word2vec_embeddings(texts, model, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = []
        for text in batch:
            words = word_tokenize(text.lower()) 
            word_vectors = [model.wv[word] for word in words if word in model.wv] 
            if word_vectors:
                batch_embeddings.append(np.mean(word_vectors, axis=0))  
            else:
                batch_embeddings.append(np.zeros(model.vector_size))  # 更稳妥地获取维度
        embeddings.extend(batch_embeddings)
    return embeddings

#Calculate Word2Vec cosine similarity
def calculate_word2vec_similarity(texts1, texts2, model,batch_size=8):
    # NaN or None
    texts1 = ["" if pd.isna(text) else text for text in texts1]
    texts2 = ["" if pd.isna(text) else text for text in texts2]

    embeddings1 = get_word2vec_embeddings(texts1,model, batch_size=batch_size)
    embeddings2 = get_word2vec_embeddings(texts2,model, batch_size=batch_size)

    similarities = []
    for v1, v2 in zip(embeddings1, embeddings2):
        if np.all(v1 == 0) or np.all(v2 == 0):  #zero vectors
            similarities.append(0)  
        else:
            similarities.append(1 - cosine(v1, v2))
    
    return similarities
