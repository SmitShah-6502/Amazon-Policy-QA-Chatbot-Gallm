## Amazon-Policy-QA-Chatbot-Gallm

A Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about Amazon policies for Buyers, Sellers, and Admins. The chatbot leverages a fine-tuned GPT-Neo model, FAISS for vector-based retrieval, and Streamlit for an interactive web interface.

#Project Overview
This project combines natural language processing (NLP), document retrieval, and a web interface to create an intelligent chatbot capable of answering Amazon policy-related questions. It extracts text from PDF documents, processes it into embeddings, retrieves relevant context using FAISS, and generates responses using a fine-tuned GPT-Neo model.

#Features
Role-Based Responses: Customizes answers based on the user's role (Buyer, Seller, Admin).
PDF Processing: Extracts and indexes text from PDF files in the data/ folder.
Retrieval-Augmented Generation (RAG): Combines document retrieval with language generation for accurate, context-aware responses.
Interactive UI: Built with Streamlit, featuring a sleek Amazon-style interface and fun animations.
CPU-Optimized: Designed to run efficiently on CPU-only environments.

#Project Flow

Data Preparation (rag.py)
PDFs in the data/ folder are processed using PyMuPDF (fitz).
Text is extracted, split into chunks (500 characters with 100-character overlap), and embedded using sentence-transformers/all-MiniLM-L6-v2.
Embeddings are indexed with FAISS and saved along with metadata (faiss_index.bin, metadata.pkl).

Model Fine-Tuning (fine_tune.py)
The GPT-Neo-1.3B model (EleutherAI/gpt-neo-1.3B) is fine-tuned on the extracted PDF text.
Training is optimized for CPU execution with a small batch size and single epoch.
The fine-tuned model is saved to ./fine_tuned_EleutherAI.

Retrieval System (qa.py, retriever.py)
FAISS index and metadata are loaded.
User queries are embedded and searched against the FAISS index to retrieve the top-k relevant text chunks.

Response Generation (gemini_key.py)
The fine-tuned GPT-Neo model generates responses based on the query, retrieved context, and user role.
Role-specific prompts ensure tailored answers.

Web Interface (app.py)
Streamlit provides an interactive UI with a role selector, query input, and response display.
Retrieved documents are shown in an expandable section, and responses are styled in a bubble format.

Requirements
Ensure you have Python 3.8+ installed. Install dependencies using the provided requirements.txt:

Install Dependencies
pip install -r requirements.txt

Prepare Data
Place Amazon policy PDFs in the data/ folder.
Process PDFs and Create FAISS Index
python rag.py

Fine-Tune the Model
Run the fine-tuning script (ensure sufficient disk space and memory):
python fine_tune.py
This creates ./fine_tuned_EleutherAI.

Run the Chatbot
streamlit run app.py
Open your browser at http://localhost:8501.

Usage
Select a Role: Choose "Buyer," "Seller," or "Admin" from the sidebar.
Ask a Question: Type your query (e.g., "What are the guidelines for selling on Amazon?") and click "Ask 🚀".
View Response: The chatbot retrieves relevant policy sections and generates a response based on your role.

Example
Query: "How can I start selling on Amazon?"
Role: Seller
Output: A detailed response outlining steps and policies for new Amazon sellers, with relevant PDF excerpts.

File Structure
amazon-policy-chatbot/
│
├── data/                   # Folder for PDF documents
├── fine_tuned_EleutherAI/  # Fine-tuned model output
├── results/                # Training checkpoints
├── app.py                  # Streamlit web interface
├── fine_tune.py            # Model fine-tuning script
├── gemini_key.py           # Response generation with fine-tuned model
├── qa.py                   # FAISS-based retrieval
├── rag.py                  # PDF processing and FAISS index creation
├── retriever.py            # General retrieval utilities
├── faiss_index.bin         # FAISS index file
├── metadata.pkl            # Metadata for indexed text
├── README.md               # This file
└── requirements.txt        # Python dependencies

Performance: The project is CPU-optimized but may require significant memory for large PDFs or during fine-tuning.
Model Path: Update MODEL_PATH in gemini_key.py if you rename the fine-tuned model folder.
Customization: Adjust top_k in qa.py or retrieve_context for more/less retrieved context.
Troubleshooting

Memory Issues: Reduce per_device_train_batch_size in fine_tune.py or process fewer PDFs.
FAISS Errors: Ensure faiss_index.bin and metadata.pkl exist and match the PDF data.
Streamlit Not Loading: Verify all dependencies are installed and run streamlit run app.py from the project root.

Future Improvements
Add GPU support for faster training and inference.
Integrate real-time web scraping for up-to-date Amazon policies.
Enhance UI with chat history and multi-turn conversations.
