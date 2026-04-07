import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# 1. Load Environment Variables (API Key from .env file)
load_dotenv()

def ingest_pdf(file_path):
    """Reads the PDF, splits it into chunks, and saves to a vector database."""
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found in the folder!")
        return None

    print(f"--- 📄 Loading {file_path} ---")
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Split into Chunks (Overlap ensures context isn't cut off)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(documents)
    print(f"--- ✅ Split into {len(chunks)} chunks ---")

    # Create Embeddings & Store in Vector DB
    # This turns text into math and saves it to 'chroma_db' folder
    print("--- 🧠 Creating Vector Database (This may take a moment)... ---")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("--- 💾 Database saved to folder 'chroma_db' ---")
    return vector_db

def ask_question(vector_db, query):
    """Uses Gemini to answer a question based ONLY on the PDF context."""
    print(f"\n--- 🤖 Asking Gemini: '{query}' ---")
    
    # Setup Gemini Model
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

    # Create the Retrieval Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )

    # Get the answer
    response = qa_chain.invoke(query)
    print("\n--- ✨ ANSWER ---")
    print(response['result'])
    print("-----------------\n")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # Ensure your PDF is named 'test.pdf' in the same folder
    pdf_file = "test.pdf" 
    
    # Step 1: Index the PDF
    db = ingest_pdf(pdf_file)
    
    if db:
        # Step 2: Ask a Question! 
        # (Change this string to whatever you want to ask your PDF)
        user_query = "Summarize the key points of this document in 3 bullet points."
        ask_question(db, user_query)