from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import os
import tempfile
import uuid
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="DocuChat AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

sessions = {}


def clean_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')


@app.get("/")
def root():
    return {"status": "DocuChat AI Backend chal raha hai ✅"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF upload kar ke process karna aur session_id return karna"""

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

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db_dir = tempfile.mkdtemp()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )

    session_id = str(uuid.uuid4())
    sessions[session_id] = {
        "vectorstore": vectorstore,
        "history": [],
        "filename": file.filename
    }

    os.unlink(tmp_path)

    return {
        "session_id": session_id,
        "filename": file.filename,
        "pages": len(pages),
        "chunks": len(chunks)
    }


@app.post("/chat")
async def chat(session_id: str = Form(...), query: str = Form(...)):
    """Ek session ke PDF se sawal poochna"""

    if session_id not in sessions:
        return {"error": "Session nahi mila. Pehle PDF upload karein."}

    session = sessions[session_id]
    vectorstore = session["vectorstore"]

    # Relevant chunks dhoondna (score ke sath)
    results = vectorstore.similarity_search_with_score(query, k=5)

    # Sirf achhi tarah se related chunks rakhna (kam distance = zyada relevant)
    relevant_docs = [doc for doc, score in results if score < 1.0]
    context = "\n\n".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant content found in the document."

    # Sources nikalna (sirf tab jab relevant content mila ho)
    sources = []
    seen_pages = set()
    for doc in relevant_docs:
        page_num = doc.metadata.get("page", None)
        if page_num is not None and page_num not in seen_pages:
            sources.append(page_num + 1)
            seen_pages.add(page_num)

    recent_history = session["history"][-6:]
    history_text = ""
    for msg in recent_history:
        role_label = "User" if msg["role"] == "user" else "Assistant"
        history_text += f"{role_label}: {msg['content']}\n"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

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

    response = llm.invoke(prompt)
    answer = response.content

    session["history"].append({"role": "user", "content": query})
    session["history"].append({"role": "assistant", "content": answer})

    return {"answer": answer, "sources": sorted(sources)}