import fitz

PDF_PATH = "documents/thirdvizion-services.pdf"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_text():
    doc = fitz.open(PDF_PATH)

    text = ""

    for page in doc:
        text += page.get_text() + "\n"

    doc.close()
    return text


def create_chunks(text):
    chunks = []

    start = 0

    while start < len(text):
        end = start + CHUNK_SIZE

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


text = extract_text()
chunks = create_chunks(text)

print(f"Total characters: {len(text)}")
print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)
