import os
import shutil
import uuid

import fitz
import chromadb

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from rag import ask_question

from dotenv import load_dotenv

# Load backend/.env (sits next to this file).
# Defaults below keep the old hardcoded behaviour
# when a variable is missing.
load_dotenv(
    os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        ".env",
    )
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

# backend/
BACKEND_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# local-rag-app/
PROJECT_ROOT = os.path.dirname(
    BACKEND_DIR
)


def _resolve_dir(value: str, default_name: str) -> str:
    # Absolute env paths are used as-is,
    # relative ones resolve against the project root.
    path = (value or default_name).strip() or default_name

    if not os.path.isabs(path):
        path = os.path.join(
            PROJECT_ROOT,
            path,
        )

    return path


# Existing project folders
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


# Make sure documents folder exists
os.makedirs(
    DOCUMENTS_DIR,
    exist_ok=True
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="ThirdVizion AI API",
    description="ThirdVizion RAG Backend API",
    version="1.0.0"
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,"
            "http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# EMBEDDING MODEL
# ============================================================

print()
print("=" * 60)
print("Loading embedding model...")
print("=" * 60)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


# ============================================================
# CHROMADB
# ============================================================

print()
print("=" * 60)
print("Connecting to ChromaDB...")
print("=" * 60)

print(f"ChromaDB path: {CHROMA_PATH}")

chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH
)

collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME
)

print("ChromaDB connected.")
print(
    f"Existing chunks: {collection.count()}"
)

print("=" * 60)


# ============================================================
# REQUEST MODEL
# ============================================================

class ChatRequest(BaseModel):
    question: str


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "status": "ok",
        "message": "ThirdVizion AI API is running"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "ThirdVizion AI API"
    }


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(
    pdf_path: str
) -> str:

    doc = fitz.open(pdf_path)

    text = ""

    try:

        for page in doc:

            page_text = page.get_text()

            if page_text:

                text += page_text + "\n"

    finally:

        doc.close()

    return text.strip()


# ============================================================
# CHUNK CREATION
# ============================================================

def create_chunks(text: str):

    chunks = []

    start = 0

    step = (
        CHUNK_SIZE
        - CHUNK_OVERLAP
    )

    while start < len(text):

        end = start + CHUNK_SIZE

        chunk = text[
            start:end
        ].strip()

        if chunk:

            chunks.append(
                chunk
            )

        start += step

    return chunks


# ============================================================
# LEGACY FILENAME RESOLUTION
# ============================================================

def resolve_orphan_filename():
    # Chunks ingested by the standalone ingest script were
    # stored without any metadata, so the original filename
    # is unknown. Attribute them to the PDF found in the
    # documents folder when possible.

    try:

        pdfs = sorted(
            f for f in os.listdir(DOCUMENTS_DIR)
            if f.lower().endswith(".pdf")
        )

    except Exception:

        pdfs = []

    if len(pdfs) == 1:

        return pdfs[0]

    if "thirdvizion-services.pdf" in pdfs:

        return "thirdvizion-services.pdf"

    if pdfs:

        return pdfs[0]

    return "legacy-documents.pdf"


# ============================================================
# FIND EXISTING DOCUMENT
# ============================================================

def find_existing_document(
    filename: str
):

    try:

        results = collection.get(
            include=["metadatas"]
        )

    except Exception:

        return None

    metadatas = results.get(
        "metadatas",
        []
    )

    target_filename = (
        filename.lower()
    )

    has_orphan_chunks = False

    for metadata in metadatas:

        if not metadata:

            has_orphan_chunks = True
            continue

        # ----------------------------------------------------
        # New document metadata
        # ----------------------------------------------------

        original_filename = (
            metadata.get(
                "original_filename"
            )
        )

        if original_filename:

            if (
                original_filename.lower()
                == target_filename
            ):

                return metadata

        # ----------------------------------------------------
        # Source metadata
        # ----------------------------------------------------

        source = metadata.get(
            "source"
        )

        if not source:
            continue

        source_filename = os.path.basename(
            source
        )

        # Exact match
        if (
            source_filename.lower()
            == target_filename
        ):

            return metadata

        # Previous version stored filename:
        # example:
        # services_a82f31c4.pdf

        source_without_ext = os.path.splitext(
            source_filename
        )[0]

        target_without_ext = os.path.splitext(
            filename
        )[0]

        if source_without_ext.lower().startswith(
            target_without_ext.lower() + "_"
        ):

            return metadata

    # ----------------------------------------------------
    # Chunks without metadata (standalone ingest script)
    # ----------------------------------------------------

    if has_orphan_chunks:

        orphan_filename = resolve_orphan_filename()

        if orphan_filename.lower() == target_filename:

            return {
                "document_id": "legacy_" + orphan_filename,
                "original_filename": orphan_filename,
                "source": orphan_filename,
            }

    return None


# ============================================================
# PROCESS PDF
# ============================================================

def process_pdf(
    pdf_path: str,
    original_filename: str,
    stored_filename: str
):

    print()
    print("=" * 60)
    print(
        f"Processing: {original_filename}"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Extract text
    # --------------------------------------------------------

    print("Extracting text...")

    text = extract_pdf_text(
        pdf_path
    )

    if not text:

        raise ValueError(
            "No readable text found in the PDF."
        )

    print(
        f"Extracted characters: {len(text)}"
    )

    # --------------------------------------------------------
    # Create chunks
    # --------------------------------------------------------

    print("Creating chunks...")

    chunks = create_chunks(
        text
    )

    if not chunks:

        raise ValueError(
            "Could not create chunks from the PDF."
        )

    print(
        f"Total chunks: {len(chunks)}"
    )

    # --------------------------------------------------------
    # Create embeddings
    # --------------------------------------------------------

    print("Creating embeddings...")

    embeddings = embedding_model.encode(
        chunks,
        normalize_embeddings=True,
        show_progress_bar=False
    ).tolist()

    # --------------------------------------------------------
    # Create document ID
    # --------------------------------------------------------

    document_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # Create chunk IDs
    # --------------------------------------------------------

    ids = []

    for index in range(
        len(chunks)
    ):

        ids.append(
            f"{document_id}_{index}"
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadatas = []

    for index in range(
        len(chunks)
    ):

        metadatas.append({

            "source":
                stored_filename,

            "original_filename":
                original_filename,

            "stored_filename":
                stored_filename,

            "document_id":
                document_id,

            "chunk_index":
                index

        })

    # --------------------------------------------------------
    # Store in ChromaDB
    # --------------------------------------------------------

    print(
        "Storing in ChromaDB..."
    )

    collection.add(

        ids=ids,

        documents=chunks,

        embeddings=embeddings,

        metadatas=metadatas

    )

    print(
        f"Stored {len(chunks)} chunks."
    )

    print(
        f"Finished processing: "
        f"{original_filename}"
    )

    print("=" * 60)

    return {

        "document_id":
            document_id,

        "filename":
            original_filename,

        "stored_filename":
            stored_filename,

        "chunks":
            len(chunks)

    }


# ============================================================
# CHAT API
# ============================================================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )

    answer = ask_question(
        question
    )

    return {
        "answer": answer
    }


# ============================================================
# UPLOAD DOCUMENT
# ============================================================

@app.post("/admin/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    # --------------------------------------------------------
    # Validate file
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file selected"
        )

    if not file.filename.lower().endswith(
        ".pdf"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )

    # Prevent path traversal
    original_filename = os.path.basename(
        file.filename
    )

    # --------------------------------------------------------
    # Duplicate check
    # --------------------------------------------------------

    existing = find_existing_document(
        original_filename
    )

    if existing:

        existing_document_id = (
            existing.get(
                "document_id"
            )
        )

        existing_filename = (
            existing.get(
                "original_filename"
            )
            or existing.get(
                "source"
            )
        )

        return {

            "status":
                "existing",

            "message":
                "This document already exists.",

            "document": {

                "document_id":
                    existing_document_id,

                "filename":
                    os.path.basename(
                        existing_filename
                    )
                    if existing_filename
                    else original_filename

            }

        }

    # --------------------------------------------------------
    # Create unique stored filename
    # --------------------------------------------------------

    filename_without_ext = os.path.splitext(
        original_filename
    )[0]

    extension = os.path.splitext(
        original_filename
    )[1]

    stored_filename = (
        f"{filename_without_ext}_"
        f"{uuid.uuid4().hex[:8]}"
        f"{extension}"
    )

    file_path = os.path.join(
        DOCUMENTS_DIR,
        stored_filename
    )

    # --------------------------------------------------------
    # Save uploaded PDF
    # --------------------------------------------------------

    try:

        with open(
            file_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to save PDF: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Process PDF
    # --------------------------------------------------------

    try:

        result = process_pdf(

            file_path,

            original_filename,

            stored_filename

        )

    except Exception as e:

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )

        raise HTTPException(
            status_code=500,
            detail=(
                f"PDF processing failed: {str(e)}"
            )
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    return {

        "status":
            "success",

        "message":
            "PDF uploaded and processed successfully",

        "document":
            result

    }


# ============================================================
# LIST DOCUMENTS
# ============================================================

@app.get("/admin/documents")
def list_documents():

    try:

        results = collection.get(
            include=["metadatas"]
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=(
                f"Failed to read documents: {str(e)}"
            )
        )

    ids = results.get(
        "ids",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    documents = {}

    orphan_chunks = 0

    # ========================================================
    # READ ALL CHUNKS
    # ========================================================

    for index, metadata in enumerate(
        metadatas
    ):

        if not metadata:

            # Chunks stored without metadata
            # (standalone ingest script).
            orphan_chunks += 1
            continue

        document_id = metadata.get(
            "document_id"
        )

        # ====================================================
        # NEW DOCUMENT
        # ====================================================

        if document_id:

            filename = (
                metadata.get(
                    "original_filename"
                )
                or metadata.get(
                    "source"
                )
                or "Unknown PDF"
            )

            if document_id not in documents:

                documents[document_id] = {

                    "document_id":
                        document_id,

                    "filename":
                        os.path.basename(
                            filename
                        ),

                    "chunks":
                        0,

                    "legacy":
                        False

                }

            documents[
                document_id
            ]["chunks"] += 1

            continue

        # ====================================================
        # OLD / LEGACY DOCUMENT
        # ====================================================

        source = metadata.get(
            "source"
        )

        if not source:
            continue

        legacy_filename = os.path.basename(
            source
        )

        legacy_id = (
            "legacy_"
            + legacy_filename
        )

        if legacy_id not in documents:

            documents[legacy_id] = {

                "document_id":
                    legacy_id,

                "filename":
                    legacy_filename,

                "chunks":
                    0,

                "legacy":
                    True

            }

        documents[
            legacy_id
        ]["chunks"] += 1

    # ====================================================
    # CHUNKS WITHOUT METADATA
    # (standalone ingest script stored no metadata,
    # so group them under the PDF in documents/)
    # ====================================================

    if orphan_chunks > 0:

        orphan_filename = resolve_orphan_filename()

        orphan_id = (
            "legacy_"
            + orphan_filename
        )

        if orphan_id not in documents:

            documents[orphan_id] = {

                "document_id":
                    orphan_id,

                "filename":
                    orphan_filename,

                "chunks":
                    0,

                "legacy":
                    True

            }

        documents[
            orphan_id
        ]["chunks"] += orphan_chunks

    return {

        "status":
            "success",

        "documents":
            list(
                documents.values()
            )

    }


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete(
    "/admin/documents/{document_id}"
)
def delete_document(
    document_id: str
):

    # ========================================================
    # LEGACY DOCUMENT DELETE
    # ========================================================

    if document_id.startswith(
        "legacy_"
    ):

        filename = document_id[
            len("legacy_"):
        ]

        results = collection.get(
            include=["metadatas"]
        )

        all_ids = results.get(
            "ids",
            []
        )

        all_metadatas = results.get(
            "metadatas",
            []
        )

        ids_to_delete = []

        orphan_ids = []

        for index, metadata in enumerate(
            all_metadatas
        ):

            if not metadata:

                # Chunk without metadata
                # (standalone ingest script).
                if index < len(all_ids):

                    orphan_ids.append(
                        all_ids[index]
                    )

                continue

            document_id_value = metadata.get(
                "document_id"
            )

            source = metadata.get(
                "source"
            )

            if not source:
                continue

            source_filename = os.path.basename(
                source
            )

            if source_filename == filename:

                if index < len(all_ids):

                    ids_to_delete.append(
                        all_ids[index]
                    )

        # ----------------------------------------------------
        # Include chunks stored without metadata
        # (standalone ingest script) when they belong to
        # this legacy document.
        # ----------------------------------------------------

        if orphan_ids:

            if filename == resolve_orphan_filename():

                ids_to_delete.extend(
                    orphan_ids
                )

        if not ids_to_delete:

            raise HTTPException(
                status_code=404,
                detail="Document not found"
            )

        # Delete chunks
        collection.delete(
            ids=ids_to_delete
        )

        # Delete physical PDF
        file_path = os.path.join(
            DOCUMENTS_DIR,
            filename
        )

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )

        return {

            "status":
                "success",

            "message":
                "Document deleted successfully",

            "document_id":
                document_id,

            "filename":
                filename,

            "deleted_chunks":
                len(ids_to_delete)

        }

    # ========================================================
    # NORMAL DOCUMENT DELETE
    # ========================================================

    results = collection.get(

        where={
            "document_id":
                document_id
        },

        include=[
            "metadatas"
        ]

    )

    ids = results.get(
        "ids",
        []
    )

    metadatas = results.get(
        "metadatas",
        []
    )

    if not ids:

        raise HTTPException(
            status_code=404,
            detail="Document not found"
        )

    original_filename = None
    stored_filename = None

    if metadatas:

        metadata = metadatas[0]

        original_filename = (
            metadata.get(
                "original_filename"
            )
        )

        stored_filename = (
            metadata.get(
                "stored_filename"
            )
            or metadata.get(
                "source"
            )
        )

    # --------------------------------------------------------
    # Delete ChromaDB chunks
    # --------------------------------------------------------

    collection.delete(
        ids=ids
    )

    # --------------------------------------------------------
    # Delete physical PDF
    # --------------------------------------------------------

    if stored_filename:

        safe_filename = os.path.basename(
            stored_filename
        )

        file_path = os.path.join(
            DOCUMENTS_DIR,
            safe_filename
        )

        if os.path.exists(
            file_path
        ):

            os.remove(
                file_path
            )

    return {

        "status":
            "success",

        "message":
            "Document deleted successfully",

        "document_id":
            document_id,

        "filename":
            original_filename
            or stored_filename,

        "deleted_chunks":
            len(ids)

    }     
