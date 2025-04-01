import faiss
import pickle
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load FAISS index and metadata
index = faiss.read_index("faiss_index.bin")
with open("metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def retrieve_context(query, top_k=3):
    """Retrieves top-k relevant text chunks from PDFs using FAISS."""
    query_embedding = np.array([embedding_model.embed_query(query)], dtype=np.float32)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i in indices[0]:
        if i < len(metadata["texts"]):
            results.append(f"📄 {metadata['sources'][i]}: {metadata['texts'][i]}")
    
    return results

if __name__ == "__main__":
    query = "What are the guidelines for selling on Amazon?"
    print("🔍 Relevant Documents:", retrieve_context(query))
