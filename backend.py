from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import List, Optional, Dict
import os
import json
import tempfile
import uuid
import shutil
import time
from datetime import datetime
import redis
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from groq import RateLimitError, APIConnectionError, APIStatusError, InternalServerError
from dotenv import load_dotenv

from auth import hash_password, verify_password, create_access_token, decode_access_token

load_dotenv()

app = FastAPI(title="DocuChat AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

CHROMA_STORAGE_DIR = "chroma_sessions"

vectorstore_cache = {}

# ---------------------------------------------------------
# REDIS SETUP
# ---------------------------------------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379))

redis_client = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    db=0,
    decode_responses=True
)

SESSIONS_INDEX_KEY = "sessions_index"


def history_key(session_id: str) -> str:
    return f"chat_history:{session_id}"


def meta_key(session_id: str) -> str:
    return f"session_meta:{session_id}"


def user_key(email: str) -> str:
    return f"user:{email}"


def user_sessions_key(user_id: str) -> str:
    return f"user_sessions:{user_id}"


# ---------------------------------------------------------
# AUTH SETUP
# ---------------------------------------------------------
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


def get_user_by_email(email: str) -> Optional[Dict]:
    raw = redis_client.get(user_key(email))
    if raw is None:
        return None
    return json.loads(raw)


def create_user(email: str, password: str) -> Dict:
    user_id = str(uuid.uuid4())
    user_data = {
        "user_id": user_id,
        "email": email,
        "hashed_password": hash_password(password),
        "created_at": datetime.now().isoformat()
    }
    redis_client.set(user_key(email), json.dumps(user_data))
    return user_data


async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict:
    """Har protected endpoint pe token check karna"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalid ya expire ho chuka hai. Dobara login karein.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None:
        raise credentials_exception

    email = payload.get("sub")
    if email is None:
        raise credentials_exception

    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception

    return user


# ---------------------------------------------------------
# PYDANTIC MODELS
# ---------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    query: str


class ChatResponse(BaseModel):
    answer: Optional[str] = None
    sources: List[int] = []
    error: Optional[str] = None


class SessionMeta(BaseModel):
    filename: str
    title: str
    db_path: str
    pages: int
    chunks: int
    created_at: str


class SignupRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    email: str


def clean_text(text_):
    return text_.encode('utf-8', 'ignore').decode('utf-8')


def make_title_from_query(query: str) -> str:
    title = query.strip().replace("\n", " ")
    if len(title) > 45:
        title = title[:45].rsplit(" ", 1)[0] + "..."
    return title if title else "New Chat"


# ---------------------------------------------------------
# REDIS HELPER FUNCTIONS (chat history)
# ---------------------------------------------------------
def get_chat_history(session_id: str) -> List[Dict]:
    raw_messages = redis_client.lrange(history_key(session_id), 0, -1)
    return [json.loads(m) for m in raw_messages]


def append_chat_message(session_id: str, role: str, content: str):
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    redis_client.rpush(history_key(session_id), json.dumps(message))


def init_chat_history(session_id: str):
    pass


# ---------------------------------------------------------
# REDIS HELPER FUNCTIONS (session metadata)
# ---------------------------------------------------------
def get_session_meta(session_id: str) -> Optional[Dict]:
    raw = redis_client.get(meta_key(session_id))
    if raw is None:
        return None
    return json.loads(raw)


def save_session_meta(session_id: str, meta: dict, user_id: str = None):
    """Session ka metadata Redis mein save karna + global index + user-specific index mein add karna"""
    redis_client.set(meta_key(session_id), json.dumps(meta))
    redis_client.sadd(SESSIONS_INDEX_KEY, session_id)
    if user_id:
        redis_client.sadd(user_sessions_key(user_id), session_id)


def delete_session_data(session_id: str, user_id: str = None):
    redis_client.delete(history_key(session_id))
    redis_client.delete(meta_key(session_id))
    redis_client.srem(SESSIONS_INDEX_KEY, session_id)
    if user_id:
        redis_client.srem(user_sessions_key(user_id), session_id)


def get_vectorstore(session_id, db_path):
    if session_id in vectorstore_cache:
        return vectorstore_cache[session_id]

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)
    vectorstore_cache[session_id] = vectorstore
    return vectorstore


# ---------------------------------------------------------
# BASIC ENDPOINTS
# ---------------------------------------------------------
@app.get("/")
def root():
    return {"status": "DocuChat AI Backend chal raha hai ✅"}


@app.get("/health")
def health_check():
    try:
        redis_client.ping()
        redis_status = "connected"
    except redis.exceptions.ConnectionError:
        redis_status = "disconnected"
    return {"status": "ok", "version": "1.2.0", "redis": redis_status}


@app.get("/me")
def get_me(current_user: Dict = Depends(get_current_user)):
    """Current logged-in user ki basic info dena (frontend sidebar ke liye)"""
    return {"email": current_user["email"], "user_id": current_user["user_id"]}


# ---------------------------------------------------------
# AUTH ENDPOINTS
# ---------------------------------------------------------
@app.post("/signup")
def signup(request: SignupRequest):
    """Naya account banana"""
    existing_user = get_user_by_email(request.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Ye email pehle se registered hai.")

    if len(request.password) < 6:
        raise HTTPException(status_code=400, detail="Password kam az kam 6 characters ka hona chahiye.")

    user = create_user(request.email, request.password)

    return {"message": "Account successfully ban gaya", "email": user["email"]}


@app.post("/login", response_model=LoginResponse)
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login kar ke JWT token lena"""
    user = get_user_by_email(form_data.username)

    if not user or not verify_password(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email ya password ghalat hai.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = create_access_token(data={"sub": user["email"]})
    return LoginResponse(access_token=access_token, email=user["email"])


# ---------------------------------------------------------
# PROTECTED ENDPOINTS (login zaroori hai)
# ---------------------------------------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...), current_user: Dict = Depends(get_current_user)):
    """PDF upload kar ke process karna aur permanent session banana (sirf logged-in user)"""

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

    meta = SessionMeta(
        filename=file.filename,
        title=file.filename,
        db_path=db_path,
        pages=len(pages),
        chunks=len(chunks),
        created_at=datetime.now().isoformat()
    )
    meta_dict = meta.dict()
    meta_dict["user_id"] = current_user["user_id"]

    save_session_meta(session_id, meta_dict, user_id=current_user["user_id"])
    init_chat_history(session_id)

    os.unlink(tmp_path)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks)
    }


@app.get("/sessions")
def list_sessions(current_user: Dict = Depends(get_current_user)):
    """Sirf LOGGED-IN user ki chats dikhana"""
    session_ids = redis_client.smembers(user_sessions_key(current_user["user_id"]))

    sessions = []
    for sid in session_ids:
        meta = get_session_meta(sid)
        if meta is None:
            continue
        sessions.append({
            "session_id": sid,
            "filename": meta.get("filename"),
            "title": meta.get("title") or meta.get("filename"),
            "created_at": meta.get("created_at")
        })

    sessions.sort(key=lambda s: s["created_at"] or "", reverse=True)
    return {"sessions": sessions}


@app.get("/sessions/{session_id}/messages")
def get_messages(session_id: str, current_user: Dict = Depends(get_current_user)):
    """Ek specific session ki poori chat history dena — sirf agar owner ho"""
    meta = get_session_meta(session_id)
    if meta is None or meta.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Ye chat aapki nahi hai.")

    history = get_chat_history(session_id)
    messages = [{"role": m["role"], "content": m["content"]} for m in history]
    return {"messages": messages}


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, current_user: Dict = Depends(get_current_user)):
    """Ek chat ko poori tarah delete karna — sirf agar owner ho"""
    meta = get_session_meta(session_id)

    if meta is None or meta.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Ye chat aapki nahi hai.")

    db_path = meta.get("db_path")
    if db_path and os.path.exists(db_path):
        shutil.rmtree(db_path, ignore_errors=True)

    vectorstore_cache.pop(session_id, None)
    delete_session_data(session_id, user_id=current_user["user_id"])

    return {"status": "deleted"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: Dict = Depends(get_current_user)):
    """Ek session ke PDF se sawal poochna — sirf agar owner ho"""

    session_id = request.session_id
    query = request.query

    meta = get_session_meta(session_id)
    if not meta:
        return ChatResponse(error="Session nahi mila.", answer=None, sources=[])

    if meta.get("user_id") != current_user["user_id"]:
        raise HTTPException(status_code=403, detail="Ye chat aapki nahi hai.")

    db_path = meta["db_path"]
    vectorstore = get_vectorstore(session_id, db_path)

    history = get_chat_history(session_id)

    if len(history) == 0:
        meta["title"] = make_title_from_query(query)
        save_session_meta(session_id, meta, user_id=current_user["user_id"])

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

    recent_messages = history[-6:]
    history_text = ""
    for msg in recent_messages:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role_label}: {msg['content']}\n"

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3)

    prompt = f"""You are a helpful assistant. A document has been provided as context below.

Instructions:
1. First, try to answer the question using ONLY the document context below.
2. If the document context contains the answer, use it and base your response on it.
3. If the document context does NOT contain the answer, then answer the question using your own general knowledge instead — do not say the information is missing, just answer normally as a helpful assistant would.
4. Consider the conversation history to understand follow-up questions.
5. Always respond in proper English language only. This means standard English words and grammar — NOT Roman Urdu (Urdu written in English letters), NOT Hindi, NOT any other language or transliteration. Even if the question is asked in Urdu, Roman Urdu, Hindi, or any mixed language, your entire answer must be written in clear, natural English.

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
            return ChatResponse(
                error="Groq API ki rate limit / quota abhi khatam ho chuki hai. "
                      "Thodi dair baad try karein, ya console.groq.com par apna usage check karein.",
                answer=None,
                sources=[]
            )

        except InternalServerError:
            if attempt < max_retries:
                time.sleep(3)
                continue
            return ChatResponse(
                error="Groq model abhi busy/unavailable hai. Thodi dair baad try karein.",
                answer=None,
                sources=[]
            )

        except APIConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return ChatResponse(
                error="Groq API se connection nahi ho pa raha. Apna internet connection check karein.",
                answer=None,
                sources=[]
            )

        except APIStatusError as e:
            return ChatResponse(error=f"Groq API error: {str(e)}", answer=None, sources=[])

        except Exception as e:
            return ChatResponse(error=f"Unexpected error: {str(e)}", answer=None, sources=[])

    if answer is None:
        return ChatResponse(error="Response nahi mil saka. Dobara try karein.", answer=None, sources=[])

    append_chat_message(session_id, "user", query)
    append_chat_message(session_id, "assistant", answer)

    return ChatResponse(answer=answer, sources=sorted(sources))