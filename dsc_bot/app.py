import streamlit as st
from langchain_chroma import Chroma
# from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Load embedding model with ollama
# embed = OllamaEmbeddings(model="nomic-embed-text")

# loading with huggingface
embed = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


# Create embeddings if chroma_db does not exist
if not os.path.exists("chroma_db"):
    import huggingface_rag


# Load Chroma DB
vector_store = Chroma(
    persist_directory="chroma_db",
    embedding_function=embed
)

# Load LLM with ollama
# llm = ChatOllama(model="llama3.2")

# load LLM with hugginface
llm = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)


def is_greeting(text: str) -> bool:

    greetings = [
        "hi",
        "hello",
        "hey",
        "good morning",
        "good afternoon",
        "good evening"
    ]

    return text.lower().strip() in greetings


def is_farewell(text: str) -> bool:

    farewells = [
        "bye",
        "goodbye",
        "see you",
        "see ya",
        "later",
        "farewell"
    ]

    return text.lower().strip() in farewells


def retriever(query: str) -> str:
    retriever_obj = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever_obj.invoke(query)  # Bug 1: must call .invoke(query), not return the retriever itself
    return "\n\n".join([doc.page_content for doc in docs])


def generate_answer(query: str, context: str) -> str:

    #  # If retrieval is weak
    # if len(context.strip()) < 50:

    #     response = llm.invoke(query)

    #     return response.content
    # messages = [       
    #     SystemMessage(content=(
    #     "You are a helpful and friendly Data Science assistant. "
    #     "Use the provided context as your primary source of truth. "
    #     "If the answer is partially available in the context, answer clearly using the available information. "
    #     "If the context does not contain enough information, politely say that you do not currently have enough knowledge to answer the question. "
    #     "Do not invent facts or provide information unrelated to the provided context.\n\n"
    #     f"Context:\n{context}"
    # )),
    #     HumanMessage(content=query)
    # ]
    # response = llm.invoke(messages)  
    # return response.content       


# If retrieval is weak
    if len(context.strip()) < 50:

        response = llm(
            query,
            max_new_tokens=150,
            do_sample=True
        )

        return response[0]["generated_text"]

    # RAG Prompt
    prompt = f"""
    You are a helpful and friendly Data Science assistant.

    Use the provided context as your primary source of truth.

    If the answer is partially available in the context,
    answer clearly using the available information.

    If the context does not contain enough information,
    politely say that you do not currently have enough
    knowledge to answer the question.

    Do not invent facts or provide information unrelated
    to the provided context.

    Context:
    {context}

    Question:
    {query}
    """

    # response = llm(
    #     prompt,
    #     max_new_tokens=250,
    #     do_sample=True
    # )

    response = llm(
        prompt,
        max_new_tokens=200,
        temperature=0.3,
        do_sample=True
)
    return response[0]["generated_text"].replace(prompt, "").strip()

    # return response[0]["generated_text"]   


# Streamlit UI
st.title("IntelliDocs")

question = st.chat_input("Ask a question")

if question:
    st.chat_message("user").write(question)


    if is_greeting(question):

        st.chat_message("assistant").write(
            "Hello! Ask me any Data Science question."
        )

    elif is_farewell(question):

        st.chat_message("assistant").write(
            "Goodbye! Keep learning and building amazing AI projects."
        )

    else:
        with st.spinner("Retrieving context..."):
            context = retriever(question)

        with st.spinner("Fetching relevant answer..."):
            answer = generate_answer(question, context)

        st.write( answer) 


