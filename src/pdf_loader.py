from pypdf import PdfReader


def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text


pdf_path = "data/documents/sample.pdf"

text = extract_text_from_pdf(pdf_path)

print("Total characters:", len(text))
print("\n--- FIRST 1000 CHARACTERS ---\n")
print(text[:1000])