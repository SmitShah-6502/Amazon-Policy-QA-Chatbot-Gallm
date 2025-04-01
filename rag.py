import os
import fitz  # PyMuPDF
import faiss
import numpy as np
import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    with fitz.open(pdf_path) as doc:
        return "\n".join([page.get_text() for page in doc])

def process_pdfs(pdf_folder="data/"):
    """Processes PDFs and creates a FAISS vector store."""
    texts, metadata = [], []
    
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            metadata.append({"source": filename})
            
            # ✅ Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            for chunk in chunks:
                texts.append((chunk, filename))

    # ✅ Convert text into embeddings
    chunk_texts = [t[0] for t in texts]
    chunk_sources = [t[1] for t in texts]
    vectors = embedding_model.embed_documents(chunk_texts)

    index = faiss.IndexFlatL2(len(vectors[0]))
    index.add(np.array(vectors, dtype=np.float32))

    # ✅ Save FAISS index and metadata
    faiss.write_index(index, "faiss_index.bin")
    with open("metadata.pkl", "wb") as f:
        pickle.dump({"texts": chunk_texts, "sources": chunk_sources}, f)

    print("✅ FAISS index created successfully!")

if __name__ == "__main__":
    process_pdfs()
