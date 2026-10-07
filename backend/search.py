import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to local ChromaDB
import os

from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        ".env",
    )
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

_CHROMA_DIR = (
    os.getenv("CHROMA_DIR", "chroma_db").strip()
    or "chroma_db"
)

if not os.path.isabs(_CHROMA_DIR):
    _CHROMA_DIR = os.path.join(PROJECT_ROOT, _CHROMA_DIR)

client = chromadb.PersistentClient(
    path=_CHROMA_DIR
)

collection = client.get_or_create_collection(
    os.getenv("COLLECTION_NAME", "thirdvizion")
)

# Customer question
question = input("Ask a question: ")

# Convert question to embedding
question_embedding = model.encode(question).tolist()

# Search similar chunks
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=3
)

print("\n===== RELEVANT DOCUMENTS =====\n")

for i, document in enumerate(results["documents"][0], start=1):
    print(f"--- Result {i} ---")
    print(document)
    print()
