import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="DocuChat AI | Chat with your PDFs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        border-radius: 8px;
        font-weight: 600;
        transition: 0.2s;
    }
    .footer-note {
        text-align: center;
        color: #64748b;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None


def load_session_messages(session_id):
    try:
        res = requests.get(f"{BACKEND_URL}/sessions/{session_id}/messages")
        if res.status_code == 200:
            return res.json()["messages"]
    except requests.exceptions.ConnectionError:
        pass
    return []


def get_all_sessions():
    try:
        res = requests.get(f"{BACKEND_URL}/sessions")
        if res.status_code == 200:
            return res.json()["sessions"]
    except requests.exceptions.ConnectionError:
        pass
    return []


# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("### 📚 DocuChat AI")
    st.caption("Retrieval-Augmented Document Assistant")
    st.divider()

    if st.button("➕ New Chat", use_container_width=True, type="primary"):
        st.session_state.session_id = None
        st.session_state.chat_history = []
        st.session_state.pdf_name = None
        st.rerun()

    st.markdown("#### 📤 Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", label_visibility="collapsed")

    if uploaded_file is not None:
        if st.button("⚡ Process Document", use_container_width=True):
            with st.spinner("Analyzing document..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    res = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if res.status_code == 200:
                        data = res.json()
                        st.session_state.session_id = data["session_id"]
                        st.session_state.pdf_name = data["filename"]
                        st.session_state.chat_history = []
                        st.success("Document ready!")
                        st.rerun()
                    else:
                        st.error(f"Backend error: {res.text}")
                except requests.exceptions.ConnectionError:
                    st.error("⚠️ Backend se connect nahi ho pa raha.")

    st.divider()
    st.markdown("#### 💬 Previous Chats")

    sessions = get_all_sessions()

    if not sessions:
        st.caption("Koi purani chat nahi hai.")

    # Har session ChatGPT jaisa ek alag, isolated conversation hai —
    # click karne par sirf USI session ki history load hoti hai (koi mix nahi)
    for s in sessions:
        display_title = s.get("title") or s["filename"]
        label = display_title[:28] + ("..." if len(display_title) > 28 else "")
        is_active = (st.session_state.session_id == s["session_id"])

        col1, col2 = st.columns([4, 1])
        with col1:
            if st.button(
                f"{'🟣' if is_active else '📄'} {label}",
                key=f"select_{s['session_id']}",
                use_container_width=True,
                type="primary" if is_active else "secondary"
            ):
                # Yahan poori tarah is session par switch hota hai:
                # session_id badalta hai aur SIRF isi session ki messages load hoti hain
                st.session_state.session_id = s["session_id"]
                st.session_state.pdf_name = s["filename"]
                st.session_state.chat_history = load_session_messages(s["session_id"])
                st.rerun()
        with col2:
            if st.button("🗑️", key=f"delete_{s['session_id']}"):
                requests.delete(f"{BACKEND_URL}/sessions/{s['session_id']}")
                if st.session_state.session_id == s["session_id"]:
                    st.session_state.session_id = None
                    st.session_state.chat_history = []
                    st.session_state.pdf_name = None
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Frontend: Streamlit · Backend: FastAPI")

# ---------- Main ----------
st.markdown('<p class="main-header">Chat with your Documents</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF and get instant, grounded answers — powered by RAG.</p>', unsafe_allow_html=True)

if st.session_state.session_id is None:
    st.markdown("""
    <div class="status-card">
        👋 <strong>Get started:</strong> Upload a PDF from the sidebar, or select a previous chat to continue.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🔍 Smart Retrieval**\n\nFinds the most relevant sections of your document.")
    with col2:
        st.info("**💾 Persistent History**\n\nAll your chats are saved and available anytime.")
    with col3:
        st.info("**🔒 Grounded Answers**\n\nResponses are based on your document's content.")

else:
    st.markdown(f"""
    <div class="status-card">
        ✅ <strong>{st.session_state.pdf_name}</strong> — ask anything below.
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
                answer = None
                sources = []
                try:
                    res = requests.post(
                        f"{BACKEND_URL}/chat",
                        data={"session_id": st.session_state.session_id, "query": query}
                    )
                    if res.status_code == 200:
                        result = res.json()
                        if result.get("error"):
                            answer = f"⚠️ {result['error']}"
                        else:
                            answer = result["answer"]
                            sources = result.get("sources", [])
                    else:
                        answer = f"⚠️ Backend error: {res.text}"
                except requests.exceptions.ConnectionError:
                    answer = "⚠️ Backend se connect nahi ho pa raha. Check karein ke uvicorn server chal raha hai."

                st.write(answer)
                if sources:
                    pages_str = ", ".join([str(p) for p in sources])
                    st.caption(f"📄 Source: Page {pages_str}")

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

        # Pehle sawal ke baad backend title generate karta hai —
        # sidebar ko turant refresh karna taake naya title turant dikhe
        st.rerun()

st.markdown('<p class="footer-note">DocuChat AI — Streamlit + FastAPI + Gemini + ChromaDB + SQLite</p>', unsafe_allow_html=True)