from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
import sqlite3
import shutil
import time
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from groq import RateLimitError, APIConnectionError, APIStatusError, InternalServerError
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DocuChat AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_FILE = "docuchat.db"
CHAT_HISTORY_CONNECTION = f"sqlite:///{DB_FILE}"  # LangChain isi database mein history save karega
CHROMA_STORAGE_DIR = "chroma_sessions"

vectorstore_cache = {}


# ---------- Database Setup (sirf sessions table — messages ab LangChain sambhalta hai) ----------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            filename TEXT,
            title TEXT,
            db_path TEXT,
            pages INTEGER,
            chunks INTEGER,
            created_at TEXT
        )
    """)
    conn.commit()

    try:
        c.execute("ALTER TABLE sessions ADD COLUMN title TEXT")
        conn.commit()
    except sqlite3.OperationalError:
        pass  # column already exists

    conn.close()


init_db()


def clean_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')


def make_title_from_query(query: str) -> str:
    """User ke pehle sawal se ek chota, saaf title banana (ChatGPT jaisa)"""
    title = query.strip().replace("\n", " ")
    if len(title) > 45:
        title = title[:45].rsplit(" ", 1)[0] + "..."
    return title if title else "New Chat"


def get_chat_history(session_id):
    """LangChain ka history object — ye khud database mein save/load karta hai"""
    return SQLChatMessageHistory(
        session_id=session_id,
        connection=CHAT_HISTORY_CONNECTION
    )


def get_vectorstore(session_id, db_path):
    """Vectorstore ko cache se ya disk se load karna"""
    if session_id in vectorstore_cache:
        return vectorstore_cache[session_id]

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    vectorstore_cache[session_id] = vectorstore
    return vectorstore


@app.get("/")
def root():
    return {"status": "DocuChat AI Backend chal raha hai ✅"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF upload kar ke process karna aur permanent session banana"""

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        content = await file.read()
        tmp_file.write(content)
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)

    for chunk in chunks:
        chunk.page_content = clean_text(chunk.page_content)

    session_id = str(uuid.uuid4())

    db_path = os.path.join(CHROMA_STORAGE_DIR, session_id)
    os.makedirs(db_path, exist_ok=True)

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )
    vectorstore_cache[session_id] = vectorstore

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "INSERT INTO sessions (session_id, filename, title, db_path, pages, chunks, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (session_id, file.filename, file.filename, db_path, len(pages), len(chunks), datetime.now().isoformat())
    )
    conn.commit()
    conn.close()

    os.unlink(tmp_path)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks)
    }


@app.get("/sessions")
def list_sessions():
    """Sab purani chats ki list dena (title ke saath, ChatGPT jaisa)"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT session_id, filename, title, created_at FROM sessions ORDER BY created_at DESC")
    rows = c.fetchall()
    conn.close()

    return {
        "sessions": [
            {
                "session_id": r[0],
                "filename": r[1],
                "title": r[2] if r[2] else r[1],
                "created_at": r[3]
            } for r in rows
        ]
    }


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """Ek specific session ki poori chat history dena — ab LangChain se"""
    history = get_chat_history(session_id)
    messages = []
    for msg in history.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})
    return {"messages": messages}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Ek chat ko poori tarah delete karna"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT db_path FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()

    if row:
        db_path = row[0]
        c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        conn.commit()

        if os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

        vectorstore_cache.pop(session_id, None)

        # LangChain history bhi clear karna
        history = get_chat_history(session_id)
        history.clear()

    conn.close()
    return {"status": "deleted"}


@app.post("/chat")
async def chat(session_id: str = Form(...), query: str = Form(...)):
    """Ek session ke PDF se sawal poochna"""

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT db_path, title FROM sessions WHERE session_id = ?", (session_id,))
    row = c.fetchone()

    if not row:
        conn.close()
        return {"error": "Session nahi mila.", "answer": None, "sources": []}

    db_path, current_title = row
    vectorstore = get_vectorstore(session_id, db_path)

    # ---------- LangChain History Load Karna ----------
    history = get_chat_history(session_id)

    # Agar ye session ka pehla sawal hai, to title update karna (ChatGPT jaisa)
    if len(history.messages) == 0:
        new_title = make_title_from_query(query)
        c.execute("UPDATE sessions SET title = ? WHERE session_id = ?", (new_title, session_id))
        conn.commit()

    conn.close()

    # Relevant chunks dhoondna
    results = vectorstore.similarity_search_with_score(query, k=5)
    relevant_docs = [doc for doc, score in results if score < 1.0]
    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant content found in the document."

    sources = []
    seen_pages = set()
    for doc in relevant_docs:
        page_num = doc.metadata.get("page", None)
        if page_num is not None and page_num not in seen_pages:
            sources.append(page_num + 1)
            seen_pages.add(page_num)

    # Pichli history (last 6 messages) ko text mein convert karna
    recent_messages = history.messages[-6:]
    history_text = ""
    for msg in recent_messages:
        role_label = "User" if isinstance(msg, HumanMessage) else "Assistant"
        history_text += f"{role_label}: {msg.content}\n"

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    prompt = f"""You are a helpful assistant. A document has been provided as context below.

Instructions:
1. First, try to answer the question using ONLY the document context below.
2. If the document context contains the answer, use it and base your response on it.
3. If the document context does NOT contain the answer, then answer the question using your own general knowledge instead — do not say the information is missing, just answer normally as a helpful assistant would.
4. Consider the conversation history to understand follow-up questions.
5. Answer in the same language the question was asked in.

Document Context:
{context}

Conversation History:
{history_text}

Current Question: {query}

Answer:"""

    # ---------- Groq API call with error handling + retry ----------
    answer = None
    max_retries = 2

    for attempt in range(max_retries + 1):
        try:
            response = llm.invoke(prompt)
            answer = response.content
            break

        except RateLimitError:
            return {
                "error": "Groq API ki rate limit / quota abhi khatam ho chuki hai. "
                         "Thodi dair baad try karein, ya console.groq.com par apna usage check karein.",
                "answer": None,
                "sources": []
            }

        except InternalServerError:
            if attempt < max_retries:
                time.sleep(3)
                continue
            return {
                "error": "Groq model abhi busy/unavailable hai. Thodi dair baad try karein.",
                "answer": None,
                "sources": []
            }

        except APIConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {
                "error": "Groq API se connection nahi ho pa raha. Apna internet connection check karein.",
                "answer": None,
                "sources": []
            }

        except APIStatusError as e:
            return {"error": f"Groq API error: {str(e)}", "answer": None, "sources": []}

        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}", "answer": None, "sources": []}

    if answer is None:
        return {"error": "Response nahi mil saka. Dobara try karein.", "answer": None, "sources": []}

    # ---------- LangChain History Mein Save Karna ----------
    history.add_user_message(query)
    history.add_ai_message(answer)

    return {"answer": answer, "sources": sorted(sources)}