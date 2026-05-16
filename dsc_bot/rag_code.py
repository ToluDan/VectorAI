from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from sentence_transformers import SentenceTransformer
from transformers import pipeline



##load the pdf
documents = PyPDFLoader("Concepts.pdf").load()

#splitting the document into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size =500, chunk_overlap = 50)
chunks = splitter.split_documents(documents)

# embed the chunks and store them in a chroma db
# embed = OllamaEmbeddings(model= "llama3.2:l atest")
embed =OllamaEmbeddings(model="nomic-embed-text")
vector_store =Chroma.from_documents(chunks, embed, persist_directory="chroma_db")

print("done")
