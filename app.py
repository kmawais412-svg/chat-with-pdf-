import streamlit as st
import requests

# ---------- Backend URL ----------
BACKEND_URL = "http://127.0.0.1:8000"

# ---------- Page Config ----------
st.set_page_config(
    page_title="DocuChat AI | Chat with your PDFs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS ----------
st.markdown("""
<style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 1.5rem;
    }
    .status-card {
        background: rgba(99, 102, 241, 0.08);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    .stButton>button {
        background: linear-gradient(90deg, #6366f1, #a855f7);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: 0.2s;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------- Session State ----------
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = None

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📚 DocuChat AI")
    st.caption("Retrieval-Augmented Document Assistant")
    st.divider()

    st.markdown("#### 📤 Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.caption(f"Selected: **{uploaded_file.name}**")
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Analyzing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{BACKEND_URL}/upload", files=files)

                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.pdf_name = data["filename"]
                        st.session_state.pdf_stats = {"pages": data["pages"], "chunks": data["chunks"]}
                        st.session_state.chat_history = []
                        st.success("Document ready!")
                        st.rerun()
                    else:
                        st.error(f"Backend error: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Backend se connect nahi ho pa raha. Kya FastAPI server chal raha hai? (uvicorn backend:app --reload)")

    st.divider()

    if st.session_state.pdf_name:
        st.markdown("#### 📊 Document Info")
        st.markdown(f"**File:** {st.session_state.pdf_name}")
        st.markdown(f"**Pages:** {st.session_state.pdf_stats['pages']}")
        st.markdown(f"**Chunks indexed:** {st.session_state.pdf_stats['chunks']}")
        st.divider()
        if st.button("🗑️ Clear & Start Over", use_container_width=True):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.pdf_name = None
            st.session_state.pdf_stats = None
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Frontend: Streamlit · Backend: FastAPI")

# ---------- Main Header ----------
st.markdown('<p class="main-header">Chat with your Documents</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF and get instant, grounded answers from its content — powered by RAG.</p>', unsafe_allow_html=True)

# ---------- Main Content ----------
if st.session_state.session_id is None:
    st.markdown("""
    <div class="status-card">
        👋 <strong>Get started:</strong> Upload a PDF from the sidebar and click <em>Process Document</em> to begin chatting.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Smart Retrieval**\n\nFinds the most relevant sections of your document for every question.")
    with col2:
        st.info("**🌐 Multilingual**\n\nAsk in English, Urdu, or any language — the assistant responds in kind.")
    with col3:
        st.info("**🔒 Grounded Answers**\n\nResponses are based only on your document's actual content.")

else:
    st.markdown(f"""
    <div class="status-card">
        ✅ <strong>{st.session_state.pdf_name}</strong> is loaded and ready — ask anything below.
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about your document...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.chat_history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/chat",
                        data={"session_id": st.session_state.session_id, "query": query}
                    )
                    if res.status_code == 200:
                        answer = res.json()["answer"]
                    else:
                        answer = f"Backend error: {res.text}"
                except requests.exceptions.ConnectionError:
                    answer = "⚠️ Backend se connect nahi ho pa raha. FastAPI server check karein."

                st.write(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

st.markdown('<p class="footer-note">DocuChat AI — Streamlit Frontend + FastAPI Backend + Gemini + ChromaDB</p>', unsafe_allow_html=True)