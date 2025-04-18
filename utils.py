# # utils.py
import fitz  # PyMuPDF
from langchain_core.documents import Document

def load_conversations_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()

    # Normalize speaker turns
    lines = text.split("\n")
    turns = []
    for line in lines:
        if line.strip().startswith("Doctor:") or line.strip().startswith("Patient:"):
            speaker, content = line.split(":", 1)
            turns.append({"speaker": speaker.strip(), "text": content.strip()})
    return turns

def chunk_conversation_turns(turns):
    chunks = []
    for i, turn in enumerate(turns):
        meta = {"speaker": turn["speaker"], "turn_id": i}
        chunks.append(Document(page_content=turn["text"], metadata=meta))
    return chunks


