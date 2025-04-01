import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load Embedding Model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_index(dataframe, column):
    """Creates FAISS index for fast text retrieval."""
    texts = dataframe[column].astype(str).tolist()
    embeddings = model.encode(texts, convert_to_numpy=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    return index, texts

def retrieve_data(query, index, texts, top_k=5):
    """Retrieves top-k similar rows."""
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    return [texts[i] for i in indices[0]]

