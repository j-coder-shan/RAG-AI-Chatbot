"""Streamlit chat UI — redesigned for ChatGPT-style aesthetic (Phase 6)."""

from __future__ import annotations

import os
import streamlit as st
import httpx

# Load configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="RAG AI Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom premium styling via single markdown injection
st.markdown(
    """
    <style>
    /* Hiding Streamlit header/footer chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Premium dark layout */
    .stApp {
        background-color: #0D0D0D !important;
        color: #F9FAFB !important;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }

    [data-testid="stSidebar"] {
        background-color: #171717 !important;
        border-right: 1px solid #2D2D2D !important;
    }

    /* Chat Messages styling override */
    div[data-testid="stChatMessage"] {
        background-color: transparent !important;
        border: none !important;
        padding: 10px 0 !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]),
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarImage-user"]) {
        flex-direction: row-reverse !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageAvatar"],
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarImage-user"]) div[data-testid="stChatMessageAvatar"] {
        display: none !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarImage-user"]) div[data-testid="stChatMessageContent"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        border-radius: 18px !important;
        padding: 12px 18px !important;
        max-width: 70% !important;
        margin-left: auto !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15) !important;
    }

    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) div[data-testid="stChatMessageContent"],
    div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarImage-assistant"]) div[data-testid="stChatMessageContent"] {
        background-color: #1E1E2E !important;
        color: #E5E7EB !important;
        border-radius: 18px !important;
        padding: 12px 18px !important;
        max-width: 80% !important;
        margin-right: auto !important;
        border: 1px solid #2D2D2D !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
    }

    /* Sidebar headers and subtitles */
    .sidebar-title {
        font-size: 22px !important;
        font-weight: 700 !important;
        color: #F9FAFB !important;
        margin-bottom: 20px !important;
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
    }

    .sidebar-section {
        font-size: 11px !important;
        font-weight: 600 !important;
        color: #6B7280 !important;
        letter-spacing: 1.5px !important;
        margin-top: 25px !important;
        margin-bottom: 10px !important;
    }

    .file-chip {
        background-color: #1E293B !important;
        border: 1px solid #3B82F6 !important;
        border-radius: 20px !important;
        padding: 6px 12px !important;
        font-size: 13px !important;
        color: #F9FAFB !important;
        display: inline-flex !important;
        align-items: center !important;
        gap: 6px !important;
        margin-top: 10px !important;
        max-width: 100% !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }

    .chip-status {
        color: #10B981 !important;
        font-weight: 600 !important;
        font-size: 11px !important;
    }

    /* Empty state welcome card */
    .empty-state {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
        justify-content: center !important;
        height: 60vh !important;
        text-align: center !important;
        color: #9CA3AF !important;
    }

    .empty-logo {
        font-size: 64px !important;
        margin-bottom: 20px !important;
    }

    .empty-title {
        font-size: 28px !important;
        font-weight: 700 !important;
        color: #F9FAFB !important;
        margin-bottom: 8px !important;
    }

    .empty-subtitle {
        font-size: 16px !important;
        color: #6B7280 !important;
    }

    /* Querying document context bar */
    .document-bar {
        background-color: #171717 !important;
        border: 1px solid #2D2D2D !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        margin-bottom: 20px !important;
        font-size: 14px !important;
        color: #9CA3AF !important;
        display: flex !important;
        align-items: center;
        gap: 8px;
    }

    /* Custom Red Outline Clear Chat Button */
    div.stButton > button {
        background-color: transparent !important;
        color: #EF4444 !important;
        border: 1px solid #EF4444 !important;
        border-radius: 8px !important;
        transition: all 0.2s ease-in-out !important;
        margin-top: 20px !important;
    }

    div.stButton > button:hover {
        background-color: #EF4444 !important;
        color: #FFFFFF !important;
        border-color: #EF4444 !important;
    }

    /* Source Expander customization */
    div[data-testid="stExpander"] {
        background-color: #1E1E2E !important;
        border: 1px solid #2D2D2D !important;
        border-radius: 8px !important;
        margin-top: 8px !important;
    }

    div[data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #3B82F6 !important;
    }

    /* Pulsing dots indicator for thinking state */
    .pulsing-dots {
        display: flex !important;
        align-items: center !important;
        color: #E5E7EB !important;
        font-weight: 500 !important;
    }

    .pulsing-dots span {
        animation: pulse 1.4s infinite both !important;
        font-size: 16px !important;
        margin: 0 2px !important;
    }

    .pulsing-dots span:nth-child(2) {
        animation-delay: .2s !important;
    }

    .pulsing-dots span:nth-child(3) {
        animation-delay: .4s !important;
    }

    @keyframes pulse {
        0% { opacity: .2; }
        20% { opacity: 1; }
        100% { opacity: .2; }
    }

    /* Warning response color styles */
    .not-found-bubble {
        color: #F59E0B !important;
        border-left: 3px solid #F59E0B !important;
        padding-left: 10px !important;
        font-weight: 500 !important;
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

# Sidebar for PDF Uploads and System Status
with st.sidebar:
    st.markdown("<div class='sidebar-title'>🤖 RAG Chatbot</div>", unsafe_allow_html=True)
    
    st.markdown("<div class='sidebar-section'>DOCUMENT</div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload PDF",
        label_visibility="collapsed",
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
                            f"Chunks: {data.get('total_chunks', data.get('chunks', 'unknown'))}"
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

    # Display styled chip when document is indexed
    if st.session_state.collection_name:
        st.markdown(
            f"<div class='file-chip'>📄 {st.session_state.current_filename} <span class='chip-status'>✓ Indexed</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("<div class='sidebar-section'>SYSTEM STATUS</div>", unsafe_allow_html=True)
    try:
        health_resp = httpx.get(f"{BACKEND_URL}/health", timeout=3.0)
        if health_resp.status_code == 200:
            st.markdown("🟢 **Backend:** Connected")
            st.markdown("🟢 **Ollama:** Ready")
        else:
            st.markdown("🔴 **Backend:** Error")
            st.markdown("🔴 **Ollama:** Offline")
    except Exception:
        st.markdown("🔴 **Backend:** Offline")
        st.markdown("🔴 **Ollama:** Offline")

    # Outline red Clear Chat button at bottom of sidebar
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# Main Interface
if not st.session_state.collection_name:
    # Empty State Welcome Screen
    st.markdown(
        """
        <div class='empty-state'>
            <div class='empty-logo'>🤖</div>
            <div class='empty-title'>Welcome to RAG Chatbot</div>
            <div class='empty-subtitle'>Upload a PDF document on the left sidebar to get started</div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    # Thin context bar showing document name
    st.markdown(
        f"<div class='document-bar'>📄 Querying: <strong>{st.session_state.current_filename}</strong></div>",
        unsafe_allow_html=True
    )

    # Render chat history
    for message in st.session_state.messages:
        avatar = None if message["role"] == "user" else "🤖"
        with st.chat_message(message["role"], avatar=avatar):
            if message["role"] == "assistant" and "I could not find" in message["content"]:
                # Amber styling for "I could not find" fallback answer
                st.markdown(
                    f"<div class='not-found-bubble'>⚠️ {message['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                # Render source chips side-by-side using columns
                sources = message["sources"]
                cols = st.columns(len(sources))
                for idx, src in enumerate(sources):
                    with cols[idx]:
                        with st.expander(f"🔍 Context {src['index'] + 1}"):
                            st.markdown(f"**Relevance:** `{src['score']:.4f}`")
                            st.caption(src["text"])

    # User Input
    disabled = st.session_state.collection_name is None
    placeholder = "Ask a question about your document..." if not disabled else "Upload a document first..."
    
    if prompt := st.chat_input(placeholder, disabled=disabled):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Call Backend chat endpoint
        with st.chat_message("assistant", avatar="🤖"):
            # Render a ChatGPT-like pulsing dots indicator for thinking state
            thinking_indicator = st.empty()
            thinking_indicator.markdown(
                "<div class='pulsing-dots'>🤖 &nbsp; <span>●</span><span>●</span><span>●</span></div>",
                unsafe_allow_html=True
            )
            
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

                # Clear thinking indicator
                thinking_indicator.empty()

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    if "I could not find" in answer:
                        st.markdown(
                            f"<div class='not-found-bubble'>⚠️ {answer}</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(answer)
                    
                    if sources:
                        # Render source chips side-by-side using columns
                        cols = st.columns(len(sources))
                        for idx, src in enumerate(sources):
                            with cols[idx]:
                                with st.expander(f"🔍 Context {src['index'] + 1}"):
                                    st.markdown(f"**Relevance:** `{src['score']:.4f}`")
                                    st.caption(src["text"])

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
                thinking_indicator.empty()
                st.error("Failed to connect to the backend server. It may be offline.")
            except Exception as e:
                thinking_indicator.empty()
                st.error(f"An error occurred: {e}")
