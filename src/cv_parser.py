from io import BytesIO

from docx import Document
from pypdf import PdfReader


def extract_text_from_uploaded_file(file):
    if file is None:
        return ""

    filename = file.name.lower()

    if filename.endswith(".txt"):
        try:
            return file.read().decode("utf-8")
        except (UnicodeDecodeError, Exception):
            return ""

    if filename.endswith(".pdf"):
        try:
            reader = PdfReader(BytesIO(file.read()))
            pages = [page.extract_text() for page in reader.pages if page.extract_text()]
            return "\n".join(pages).strip()
        except Exception:
            return ""

    if filename.endswith(".docx"):
        try:
            doc = Document(BytesIO(file.read()))
            lines = [p.text for p in doc.paragraphs if p.text.strip()]
            return "\n".join(lines).strip()
        except Exception:
            return ""

    return ""