import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os
from model_similarity.fastText import get_fast_text_embedding
from model_similarity.sentenceTransformer import get_sentence_transformer_embedding
from model_similarity.openAI import  get_openai_embedding
from model_similarity.wordToVec import get_word2vec_embeddings

def get_embeddings(texts,model_type):
    if model_type == 'fast_text':
        return [get_fast_text_embedding(text) for text in texts]
    elif model_type == 'word2vec':
        return get_word2vec_embeddings(texts)
    elif model_type == 'sentence_transformer':
        return get_sentence_transformer_embedding(texts)
    elif model_type == 'openai':
        return get_openai_embedding(texts)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def process_jira_embeddings(df, text_column, embedding_column_name, embeddings_dir,model_type):
    
    os.makedirs(embeddings_dir, exist_ok=True)

    # Get embeddings using the appropriate model
    embeddings = get_embeddings(df[text_column].tolist(),model_type)
    
    # Create dataframe for embeddings
    #if model_type == 'openai':
       # embedding_df = pd.DataFrame({embedding_column_name: embeddings.tolist()}, index=df.index)
    #else:
    embedding_df = pd.DataFrame({embedding_column_name: embeddings}, index=df.index)
    # Concatenate with necessary columns
    embedding_df = pd.concat([df[['Issue_key_jira', 'Jira_created_date']], embedding_df], axis=1)
    
    # Save results in pickle
    embedding_df.to_pickle(path=f"{embeddings_dir}/embedding_{text_column}.pkl")
    
    # Save results in Excel
    #embedding_df.to_excel(f"{embeddings_dir}/embedding_{text_column}.xlsx", index=False)
    return embedding_df

def process_svn_embeddings(df, text_column, embedding_column_name, embeddings_dir,model_type):

    # Get embeddings using the appropriate model
    embeddings = get_embeddings(df[text_column].tolist(),model_type)
    
    # Create dataframe for embeddings
    #embedding_df = pd.DataFrame({embedding_column_name: embeddings}, index=df.index)
    embedding_df = pd.DataFrame({embedding_column_name: [list(e) for e in embeddings]}, index=df.index)
    
    # Concatenate with necessary columns 
    embedding_df = pd.concat([df[['commit_hash', 'Commit_date']], embedding_df], axis=1)

    # Save results in pickle
    embedding_df.to_pickle(path=f"{embeddings_dir}/embedding_{text_column}.pkl")
    
    # Save results in Excel
    #embedding_df.to_excel(f"{embeddings_dir}/embedding_{text_column}.xlsx", index=False)
    return embedding_df


def calculate_similarity(embedding1, embedding2):
    embedding1 = np.array(embedding1)
    embedding2 = np.array(embedding2)

    # Check for zero vectors
    if np.all(embedding1 == 0) or np.all(embedding2 == 0):
        return 0.0  # Return 0 if either embedding is a zero vector

    similarity = 1 - cosine(embedding1, embedding2)
    return similarity


def f5_log_and_summary(features_df,cartesian_df,feature_dir,model_type):

    features_df["f5_log_and_summary"]=cartesian_df.apply(lambda row: calculate_similarity(row['summary_embedding'], row['Message_embedding']), axis=1)

    #Save results in pickle
    features_df.to_pickle(path= f"{feature_dir}/features_{model_type}.pkl")
    #features_df.to_excel(f"{feature_dir}/features_{model_type}.xlsx", index=False)
    return features_df

def f6_log_and_description(features_df,cartesian_df,feature_dir,model_type):
    features_df["f6_log_and_description"]= cartesian_df.apply(
    lambda row: calculate_similarity(row['description_embedding'], row['Message_embedding']), axis=1
    )
    #Save results in pickle
    features_df.to_pickle(path= f"{feature_dir}/features_{model_type}.pkl")
    #features_df.to_excel(f"{feature_dir}/features_{model_type}.xlsx", index=False)
    return features_df


def f7_log_and_jira_all(features_df,cartesian_df,feature_dir,model_type):
    #Calculate cosine similarity for each trace
    features_df["f7_log_and_jira_all"] = cartesian_df.apply(
    lambda row: calculate_similarity(row['jira_natual_text_embedding'], row['Message_embedding']), axis=1
    )

    #Save results in pickle
    features_df.to_pickle(path= f"{feature_dir}/features_{model_type}.pkl")
    #features_df.to_excel(f"{feature_dir}/features_{model_type}.xlsx", index=False)
    return features_df