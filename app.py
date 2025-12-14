from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ================================
# 1. LOAD RESUME PDF
# ================================
loader = PyPDFLoader("resume.pdf")
documents = loader.load()

print("Resume loaded successfully!")
print(f"Total pages: {len(documents)}")

# ================================
# 2. SPLIT TEXT INTO CHUNKS
# ================================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)
print(f"Total chunks created: {len(chunks)}")

# ================================
# 3. CREATE EMBEDDINGS (FREE)
# ================================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embeddings created!")

# ================================
# 4. CREATE VECTOR STORE
# ================================
vectorstore = FAISS.from_documents(chunks, embeddings)
print("Vector store ready!")

# ================================
# 5. ASK QUESTION (RESUME SCREENING)
# ================================
# ================================
# 5. ASK QUESTION (USER INPUT)
# ================================
while True:
    query = input("\nAsk a question about the resume (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting Resume Chatbot 👋")
        break

    docs = vectorstore.similarity_search(query, k=3)

    print("\n--- Relevant Resume Sections ---\n")
    for i, doc in enumerate(docs, start=1):
        print(f"Chunk {i}:\n{doc.page_content}\n")


# ================================
# 6. SIMPLE ANALYSIS OUTPUT
# ================================
context = "\n".join(doc.page_content for doc in docs)

print("========== RESUME ANALYSIS ==========")
print("Based on the resume content:")
print(context)
