import os

import fitz
import chromadb
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

load_dotenv(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        ".env",
    )
)

# backend/
BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# local-rag-app/
PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)


def _resolve_dir(value: str, default_name: str) -> str:
    path = (value or default_name).strip() or default_name

    if not os.path.isabs(path):
        path = os.path.join(
            PROJECT_ROOT,
            path,
        )

    return path


DOCUMENTS_DIR = _resolve_dir(
    os.getenv("DOCUMENTS_DIR"),
    "documents",
)

CHROMA_PATH = _resolve_dir(
    os.getenv("CHROMA_DIR"),
    "chroma_db",
)

COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "thirdvizion",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

CHUNK_SIZE = int(
    os.getenv(
        "CHUNK_SIZE",
        "1000",
    )
)

CHUNK_OVERLAP = int(
    os.getenv(
        "CHUNK_OVERLAP",
        "200",
    )
)

PDF_PATH = os.path.join(
    DOCUMENTS_DIR,
    "thirdvizion-services.pdf",
)

# Load embedding model
print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL)

# Extract PDF text
print("Reading PDF...")
doc = fitz.open(PDF_PATH)

text = ""

for page in doc:
    text += page.get_text() + "\n"

doc.close()

# Create chunks
chunks = []

start = 0

while start < len(text):
    end = start + CHUNK_SIZE

    chunk = text[start:end].strip()

    if chunk:
        chunks.append(chunk)

    start += CHUNK_SIZE - CHUNK_OVERLAP

print(f"Created {len(chunks)} chunks")

# Create local ChromaDB
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME
)

# Create embeddings
print("Creating embeddings...")

embeddings = model.encode(chunks).tolist()

# Store in ChromaDB
collection.add(
    ids=[f"chunk-{i}" for i in range(len(chunks))],
    documents=chunks,
    embeddings=embeddings
)

print("✅ Documents stored successfully!")
print(f"Total vectors: {collection.count()}")
