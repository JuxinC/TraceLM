import openai
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine
import os
import time
import tiktoken

# Set OpenAI API key
client = OpenAI(api_key='your-api-key-here')

# Function to get the embedding of a text using OpenAI API

def get_openai_embedding(texts):
    
    embeddings = []
    model = "text-embedding-3-small"
    embedding_size = 1536   #1536 for text-embedding-3-small，text-embedding-ada-002 ｜ 3072 for text-embedding-3-large

    tokenizer = tiktoken.encoding_for_model(model)
    max_tokens = 8192

    # NaN 或 None
    texts = [" " if pd.isna(text) else text for text in texts]

    for text in texts:
        # text is blank, empty string
        if not text.strip():
            embeddings.append(np.zeros(embedding_size))
            continue

        #Truncating long text
        token_ids = tokenizer.encode(text)
        if len(token_ids) > max_tokens:
            token_ids = token_ids[:max_tokens]
            text = tokenizer.decode(token_ids)

        try:
            response = client.embeddings.create(input=[text], model=model)
            if response and hasattr(response, "data") and response.data:
                emb = response.data[0].embedding
                if np.linalg.norm(emb) < 1e-6:
                    embeddings.append(np.zeros(embedding_size))
                else:
                    embeddings.append(emb)
            else:
                raise ValueError("API response missing data")
        except Exception as e:
            print(f"Error for text: '{text}': {e}, using zero vector.")
            embeddings.append(np.zeros(embedding_size))

        time.sleep(0.5)  # Avoid triggering API rate limits

    return embeddings