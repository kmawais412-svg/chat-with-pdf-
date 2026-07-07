from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import tempfile
import uuid
import shutil
import time
from datetime import datetime
from sqlalchemy import create_engine, text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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
CHAT_HISTORY_CONNECTION = f"sqlite:///{DB_FILE}"  # LangChain isi database mein sab kuch save karega
CHROMA_STORAGE_DIR = "chroma_sessions"

# LangChain jo table khud banata hai (message_store) usay directly query karne ke liye engine
engine = create_engine(CHAT_HISTORY_CONNECTION)

vectorstore_cache = {}


def clean_text(text_):
    return text_.encode('utf-8', 'ignore').decode('utf-8')


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


def get_session_meta(session_id):
    """Session ka metadata LangChain history mein save kiye gaye System message se nikalna"""
    history = get_chat_history(session_id)
    system_messages = [m for m in history.messages if isinstance(m, SystemMessage)]
    if not system_messages:
        return None
    return json.loads(system_messages[-1].content)


def save_session_meta(session_id, meta: dict):
    """Session ka metadata System message ke roop mein LangChain history mein save karna"""
    history = get_chat_history(session_id)
    history.add_message(SystemMessage(content=json.dumps(meta)))


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


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "1.0.1"}


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

    save_session_meta(session_id, {
        "filename": file.filename,
        "title": file.filename,
        "db_path": db_path,
        "pages": len(pages),
        "chunks": len(chunks),
        "created_at": datetime.now().isoformat()
    })

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
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT session_id FROM message_store"))
        session_ids = [row[0] for row in result]

    sessions = []
    for sid in session_ids:
        meta = get_session_meta(sid)
        if meta:
            sessions.append({
                "session_id": sid,
                "filename": meta.get("filename"),
                "title": meta.get("title") or meta.get("filename"),
                "created_at": meta.get("created_at")
            })

    sessions.sort(key=lambda s: s["created_at"] or "", reverse=True)
    return {"sessions": sessions}


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    """Ek specific session ki poori chat history dena — ab LangChain se"""
    history = get_chat_history(session_id)
    messages = []
    for msg in history.messages:
        if isinstance(msg, SystemMessage):
            continue  # ye sirf metadata hai, chat mein nahi dikhana
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        messages.append({"role": role, "content": msg.content})
    return {"messages": messages}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    """Ek chat ko poori tarah delete karna"""
    meta = get_session_meta(session_id)

    if meta:
        db_path = meta.get("db_path")
        if db_path and os.path.exists(db_path):
            shutil.rmtree(db_path, ignore_errors=True)

    vectorstore_cache.pop(session_id, None)

    # LangChain history (aur usmein maujood metadata) bhi clear karna
    history = get_chat_history(session_id)
    history.clear()

    return {"status": "deleted"}


@app.post("/chat")
async def chat(session_id: str = Form(...), query: str = Form(...)):
    """Ek session ke PDF se sawal poochna"""

    meta = get_session_meta(session_id)
    if not meta:
        return {"error": "Session nahi mila.", "answer": None, "sources": []}

    db_path = meta["db_path"]
    vectorstore = get_vectorstore(session_id, db_path)

    # ---------- LangChain History Load Karna ----------
    history = get_chat_history(session_id)
    real_messages = [m for m in history.messages if not isinstance(m, SystemMessage)]

    # Agar ye session ka pehla sawal hai, to title update karna (ChatGPT jaisa)
    if len(real_messages) == 0:
        meta["title"] = make_title_from_query(query)
        save_session_meta(session_id, meta)

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
    recent_messages = real_messages[-6:]
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

    history.add_user_message(query)
    history.add_ai_message(answer)

    return {"answer": answer, "sources": sorted(sources)}