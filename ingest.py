from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ correct modern import
import os

# --- Configuration ---
DATA_PATH = "data/medical_data.txt"  
VECTORSTORE_DIR = "vectorstore"

# --- Step 1: Load Medical Knowledge File ---
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Medical data file not found at: {DATA_PATH}")

print("Loading medical data...")
loader = TextLoader(DATA_PATH)
documents = loader.load()

# --- Step 2: Split Text into Chunks ---
print("Splitting documents into manageable chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# --- Step 3: Generate Embeddings ---
print("Generating text embeddings using Ollama (nomic-embed-text)...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# --- Step 4: Create FAISS Vectorstore ---
print("Creating FAISS vectorstore...")
vectorstore = FAISS.from_documents(docs, embeddings)

# --- Step 5: Save Vectorstore ---
os.makedirs(VECTORSTORE_DIR, exist_ok=True)
vectorstore.save_local(VECTORSTORE_DIR)

print("✅ Vectorstore created and saved successfully at 'vectorstore/'!")
