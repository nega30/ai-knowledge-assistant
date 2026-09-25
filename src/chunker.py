from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


def split_text(text, chunk_size=500, overlap=100):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# PDF path
pdf_path = "data/documents/sample.pdf"

# Step 1: Extract text
text = extract_text_from_pdf(pdf_path)

print("Total characters:", len(text))

# Step 2: Split text
chunks = split_text(text)

print("Number of chunks:", len(chunks))


# Step 3: Display chunks
for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i + 1} ---")
    print(chunk)