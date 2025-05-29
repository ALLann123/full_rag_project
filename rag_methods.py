#!/usr/bin/python3
import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path 
from time import time

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings 

load_dotenv()

DB_DOCS_LIMIT = 10

def stream_llm_response(llm_stream, messages):
    response_message = ""
    for chunk in llm_stream.stream(messages):
        response_message += chunk.content
        yield chunk
    st.session_state.messages.append({"role": "assistant", "content": response_message})

def load_doc_to_db(uploaded_files):
    docs = []
    sources = []
    for doc_file in uploaded_files:
        if doc_file.name not in st.session_state.rag_sources:
            if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
                os.makedirs("source_files", exist_ok=True)
                file_path = f"./source_files/{doc_file.name}"
                with open(file_path, "wb") as file:
                    file.write(doc_file.read())

                try:
                    if doc_file.type == "application/pdf":
                        loader = PyPDFLoader(file_path)
                    elif doc_file.name.endswith(".docx"):
                        loader = Docx2txtLoader(file_path)
                    elif doc_file.type in ["text/plain", "text/markdown"]:
                        loader = TextLoader(file_path)
                    else:
                        st.warning(f"Document type {doc_file.type} not supported.")
                        continue

                    loaded_docs = loader.load()
                    docs.extend(loaded_docs)
                    sources.append(doc_file.name)

                except Exception as e:
                    st.toast(f"Error loading document {doc_file.name}: {e}", icon="⚠️")
                finally:
                    if os.path.exists(file_path):
                        os.remove(file_path)
            else:
                st.error("Maximum number of documents reached (10)")

    if docs:
        vector_db = _split_and_load_docs(docs)
        st.toast(f"Documents loaded successfully: {', '.join(sources)}", icon="✅")
        return vector_db, sources
    return None, []

def load_url_to_db(url):
    if url not in st.session_state.rag_sources:
        if len(st.session_state.rag_sources) < DB_DOCS_LIMIT:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                if docs:
                    vector_db = _split_and_load_docs(docs)
                    st.toast(f"URL content loaded successfully: {url}", icon="✅")
                    return vector_db, [url]
            except Exception as e:
                st.error(f"Error loading document from {url}: {e}")
    return None, []

def _split_and_load_docs(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=5000,
        chunk_overlap=1000,
    )
    document_chunks = text_splitter.split_documents(docs)

    if st.session_state.vector_db is None:
        st.session_state.vector_db = initialize_vector_db(document_chunks)
    else:
        st.session_state.vector_db.add_documents(document_chunks)
    
    return st.session_state.vector_db

def initialize_vector_db(document_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # Create persistent directory
    persist_dir = f"./chroma_db_{st.session_state.session_id}"
    os.makedirs(persist_dir, exist_ok=True)

    vector_db = Chroma.from_documents(
        documents=document_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    
    return vector_db

def _get_context_retriever_chain(vector_db, llm):
    retriever = vector_db.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up to get information relevant to the conversation.")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(llm):
    retriever_chain = _get_context_retriever_chain(st.session_state.vector_db, llm)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant. Answer user's questions using the provided context when available.
         Context: {context}"""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def stream_llm_rag_response(llm_stream, messages):
    conversation_rag_chain = get_conversational_rag_chain(llm_stream)
    response = conversation_rag_chain.invoke({
        "messages": messages[:-1],
        "input": messages[-1].content
    })
    response_message = f"*(RAG Response)*\n{response['answer']}"
    st.session_state.messages.append({"role": "assistant", "content": response_message})
    yield response['answer']