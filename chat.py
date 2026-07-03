from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

print("📄 Chat with PDF shuru! (bahar nikalne ke liye 'exit' likhein)\n")

while True:
    query = input("Aap: ")
    if query.lower() == "exit":
        print("Chat khatam. Allah Hafiz!")
        break

    results = vectorstore.similarity_search(query, k=5)
    context = "\n\n".join([doc.page_content for doc in results])

    # Debug: dekhein kya mila (pehle 200 characters har chunk ke)
    print(f"\n[Debug: {len(results)} chunks mile]")
    for i, doc in enumerate(results):
        print(f"--- Chunk {i+1} preview ---")
        print(doc.page_content[:200])
    print()

    # Ab prompt mein sakhti nahi rakhi, user jis language mein poochay usi mein jawab dega
    prompt = f"""Answer the question based ONLY on the context below. 
Answer in the same language the question was asked in.
If the answer is not in the context, say "This information is not in the PDF."

Context:
{context}

Question: {query}

Answer:"""

    response = llm.invoke(prompt)
    print(f"AI: {response.content}\n")