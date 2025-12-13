#ingestion/pdf_ingestion.py
import PyPDF2

def load_pdf(path):
    text = ""

    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() + "\n"

    return [{
        "source": path,
        "type": "pdf",
        "content": text
    }]
