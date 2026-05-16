# IntelliDocs AI

IntelliDocs AI is a Retrieval-Augmented Generation (RAG) chatbot built using Hugging Face models, semantic vector search, and document retrieval techniques.

The application processes PDF documents, generates embeddings, stores them in a vector database, retrieves relevant context, and generates intelligent responses using an open-source Large Language Model (LLM) — without paid AI APIs.

---

## Features

- PDF ingestion and processing
- Semantic search using vector embeddings
- Retrieval-Augmented Generation (RAG)
- Hugging Face transformer models
- ChromaDB vector storage
- Streamlit chatbot interface
- Context-aware question answering
- Greeting and farewell handling
- Hybrid retrieval + generation pipeline
- Open-source AI stack

---

## Tech Stack

### Frontend
- Streamlit

### Backend
- Python

### AI / Machine Learning
- Hugging Face Transformers
- TinyLlama
- LangChain
- sentence-transformers
- all-MiniLM-L6-v2 embeddings

### Vector Database
- ChromaDB

### Document Processing
- PyPDFLoader
- RecursiveCharacterTextSplitter

### Deployment
- Hugging Face Spaces

---

## Architecture

```text
PDF Documents
      ↓
Document Loader
      ↓
Chunking
      ↓
Embeddings Generation
      ↓
ChromaDB Vector Store
      ↓
Retriever
      ↓
TinyLlama
      ↓
Streamlit Chat Interface

## Author

### Tolulope Ajidahun
AI Engineer | Python Developer | Data Science 

🔗 LinkedIn:  
https://www.linkedin.com/in/tolulope-daniel-75141018b/

💻 GitHub:  
https://github.com/ToluDan

---

## License

This project is open-source and available under the MIT License.