from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

loader = PyPDFLoader("sample.pdf")
pages = loader.load()
print(f"Total pages loaded: {len(pages)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(pages)
print(f"Total chunks banay: {len(chunks)}")

# Kharab/corrupt characters ko saaf karna
def clean_text(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

for chunk in chunks:
    chunk.page_content = clean_text(chunk.page_content)

print("Embeddings ban rahi hain...")
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

print("Vector database ban raha hai...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ Vector database ban gaya aur save ho gaya!")