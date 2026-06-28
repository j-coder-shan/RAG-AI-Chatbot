"""Streamlit chat UI — implemented in Phase 6."""

from __future__ import annotations

import os
import streamlit as st
import httpx

# Load configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling via markdown injection
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        color: #e0e0e0;
    }
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .source-block {
        border-left: 3px solid #3b82f6;
        padding-left: 10px;
        margin-bottom: 10px;
        background-color: #1e293b;
        border-radius: 4px;
        padding-top: 5px;
        padding-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = []
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "current_filename" not in st.session_state:
    st.session_state.current_filename = None

# Sidebar for PDF Uploads
with st.sidebar:
    st.image(
        "https://img.icons8.com/color/96/adobe-acrobat-reader.png",
        width=80,
    )
    st.header("Upload Document")
    st.write("Upload a PDF document to parse, chunk, and embed it for Q&A.")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a standard PDF document (scanned/empty files will be rejected).",
    )

    if uploaded_file is not None:
        if st.session_state.current_filename != uploaded_file.name:
            with st.spinner("Indexing PDF... Please wait"):
                try:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    }
                    response = httpx.post(
                        f"{BACKEND_URL}/upload",
                        files=files,
                        timeout=120.0,  # Generous timeout for processing
                    )

                    if response.status_code == 200:
                        data = response.json()
                        st.session_state.collection_name = data["collection_name"]
                        st.session_state.current_filename = uploaded_file.name
                        # Reset chat when a new document is successfully indexed
                        st.session_state.messages = []
                        st.success(
                            f"Successfully indexed document! "
                            f"Total chunks: {data.get('total_chunks', data.get('chunks', 'unknown'))}"
                        )
                    else:
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            error_detail = response.text
                        st.error(f"Upload Failed: {error_detail}")
                        st.session_state.collection_name = None
                        st.session_state.current_filename = None

                except httpx.ConnectError:
                    st.error(
                        f"Unable to connect to the backend at {BACKEND_URL}. "
                        "Please verify the FastAPI backend server is running."
                    )
                    st.session_state.collection_name = None
                    st.session_state.current_filename = None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.collection_name = None
                    st.session_state.current_filename = None
    else:
        # If user cleared the file uploader, reset the state
        if st.session_state.current_filename is not None:
            st.session_state.collection_name = None
            st.session_state.current_filename = None
            st.session_state.messages = []

    st.divider()
    st.write("### System Status")
    try:
        health_resp = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        if health_resp.status_code == 200:
            st.status("Backend: Connected", state="complete")
        else:
            st.status("Backend: Error", state="error")
    except Exception:
        st.status("Backend: Offline", state="error")


# Main Interface
st.title("RAG AI Chatbot")
st.subheader("Interactive Document Q&A System")

if not st.session_state.collection_name:
    st.info(
        "👋 Welcome! Please upload a PDF document in the sidebar to begin "
        "asking questions."
    )
else:
    st.caption(f"Currently querying: `{st.session_state.current_filename}`")

    # Render chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and message.get("sources"):
                with st.expander("Show retrieved sources"):
                    for idx, src in enumerate(message["sources"]):
                        st.markdown(
                            f"<div class='source-block'>"
                            f"<strong>Source Chunk {src['index'] + 1}</strong> "
                            f"(Relevance Score: {src['score']:.4f})<br/>"
                            f"{src['text']}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

    # User Input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call Backend chat endpoint
        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    payload = {
                        "question": prompt,
                        "collection_name": st.session_state.collection_name,
                    }
                    response = httpx.post(
                        f"{BACKEND_URL}/chat",
                        json=payload,
                        timeout=60.0,
                    )

                    if response.status_code == 200:
                        data = response.json()
                        answer = data["answer"]
                        sources = data.get("sources", [])

                        st.markdown(answer)
                        if sources:
                            with st.expander("Show retrieved sources"):
                                for idx, src in enumerate(sources):
                                    st.markdown(
                                        f"<div class='source-block'>"
                                        f"<strong>Source Chunk {src['index'] + 1}</strong> "
                                        f"(Relevance Score: {src['score']:.4f})<br/>"
                                        f"{src['text']}"
                                        f"</div>",
                                        unsafe_allow_html=True,
                                    )

                        # Append to chat history
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    else:
                        try:
                            error_detail = response.json().get("detail", response.text)
                        except Exception:
                            error_detail = response.text
                        st.error(f"Error: {error_detail}")
                except httpx.ConnectError:
                    st.error("Failed to connect to the backend server. It may be offline.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
