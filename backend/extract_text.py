import fitz

pdf_path = "documents/thirdvizion-services.pdf"

doc = fitz.open(pdf_path)

for page_number, page in enumerate(doc, start=1):
    text = page.get_text()

    print(f"\n--- PAGE {page_number} ---\n")
    print(text)

doc.close()
