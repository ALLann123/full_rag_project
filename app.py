#!/usr/bin/python3
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from dotenv import load_dotenv
import uuid

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, AIMessage
from rag_methods import (
    stream_llm_response,
    load_doc_to_db,
    load_url_to_db,
    stream_llm_rag_response,
)

# Load environment variables
load_dotenv()

# API keys
gpt_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GITHUB_TOKEN")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize LLMs
llm_groq = ChatGroq(
    model="llama3-70b-8192",
    api_key=groq_api_key,
    temperature=0.3,
    streaming=True,
)

llm_gpt = ChatOpenAI(
    model="gpt-4o",
    temperature=0.3,
    openai_api_key=gpt_api_key,
    streaming=True
)

MODELS = {
    "GPT-4o": llm_gpt,
    "LLaMA3-70B": llm_groq
}

# Streamlit page config
st.set_page_config(
    page_title="RAG LLM app",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.markdown(
    '<h2 style="text-align: center;">📜🔎<i> Do our LLM even RAG bro?</i>🤖📰</h2>',
    unsafe_allow_html=True
)

# Session state setup
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_sources" not in st.session_state:
    st.session_state.rag_sources = []

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hi there! How can I assist you today?"
    }]

if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

if "use_rag" not in st.session_state:
    st.session_state.use_rag = False

# Sidebar UI
with st.sidebar:
    st.divider()
    model_name = st.selectbox(
        "🤖 Select a Model",
        options=list(MODELS.keys()),
        index=0
    )

    is_vector_db_loaded = st.session_state.vector_db is not None
    st.session_state.use_rag = st.toggle(
        "Use RAG",
        value=st.session_state.use_rag,
        disabled=not is_vector_db_loaded,
    )

    cols = st.columns(2)
    with cols[1]:
        st.button("Clear Chat", on_click=lambda: st.session_state.messages.clear(), type="primary")

    st.divider()
    st.header("RAG Sources:")

    uploaded_files = st.file_uploader(
        "📄 Upload documents",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
        key="file_uploader"
    )

    if uploaded_files:
        try:
            db, sources = load_doc_to_db(uploaded_files)
            if db is not None:
                st.session_state.vector_db = db
                st.session_state.rag_sources.extend(sources)
                st.session_state.use_rag = True
                st.rerun()
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

    url = st.text_input("🌐 Introduce a URL", placeholder="https://example.com", key="rag_url")
    if url:
        try:
            db, sources = load_url_to_db(url)
            if db is not None:
                st.session_state.vector_db = db
                st.session_state.rag_sources.extend(sources)
                st.session_state.use_rag = True
                st.rerun()
        except Exception as e:
            st.error(f"Error loading URL: {str(e)}")

    with st.expander(f"📚 Documents in DB ({len(st.session_state.rag_sources)})"):
        st.write(st.session_state.rag_sources)

selected_llm = MODELS[model_name]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle chat input
if prompt := st.chat_input("Your Message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        messages = [
            HumanMessage(content=m["content"]) if m["role"] == "user"
            else AIMessage(content=m["content"])
            for m in st.session_state.messages
        ]

        if not st.session_state.use_rag or st.session_state.vector_db is None:
            st.write_stream(stream_llm_response(selected_llm, messages))
        else:
            st.write_stream(stream_llm_rag_response(selected_llm, messages))