# create_vector_db.py
import json
import re
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
import os


# Function to parse the extracted text into structured sections
def parse_clinical_summary(raw_text):
    """Parse the clinical summary text into structured sections"""
    sections = {
        "Active Symptoms": "",
        "Negative Findings": "",
        "Diagnostic Conclusions": "",
        "Therapeutic Interventions": "",
        "Diagnostic Evidence": "",
        "Chronic Conditions": "",
        "Follow-up Plan": "",
        "Visit Timeline": "",
        "Summary Narrative": ""
    }

    current_section = None

    for line in raw_text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # Check if line is a section header
        is_header = False
        for section in sections.keys():
            if line.startswith(section + ":") or line == section:
                current_section = section
                is_header = True
                break

        if is_header:
            continue

        # Add content to current section
        if current_section:
            sections[current_section] += line + " "

    # Trim whitespace
    for section in sections:
        sections[section] = sections[section].strip()

    return sections


def create_documents_from_json(json_file):
    """Create Langchain documents from extracted JSON data"""
    with open(json_file, 'r') as f:
        data = json.load(f)

    documents = []

    for entry in data:
        # Parse the raw text into sections
        sections = parse_clinical_summary(entry['raw_text'])

        # Extract metadata
        metadata = entry['metadata']
        metadata['source_file'] = entry['source_file']

        # Create separate documents for each section for more granular retrieval
        for section_name, section_content in sections.items():
            if section_content:
                doc = Document(
                    page_content=section_content,
                    metadata={
                        "source": entry['source_file'],
                        "section": section_name,
                        "gender": metadata.get('gender'),
                        "age": metadata.get('age'),
                        "mrn": metadata.get('mrn'),
                        "diagnosis": metadata.get('diagnosis')
                    }
                )
                documents.append(doc)

        # Also create a document with all content for full-text searches
        full_content = "\n".join([f"{k}: {v}" for k, v in sections.items() if v])
        doc = Document(
            page_content=full_content,
            metadata={
                "source": entry['source_file'],
                "section": "FULL_TEXT",
                "gender": metadata.get('gender'),
                "age": metadata.get('age'),
                "mrn": metadata.get('mrn'),
                "diagnosis": metadata.get('diagnosis')
            }
        )
        documents.append(doc)

    return documents


def create_vector_db(documents, persist_directory="./medical_vectordb"):
    """Create a Chroma vector database from documents"""
    # Create embeddings using Ollama
    embeddings = OllamaEmbeddings(model="llama3.2")

    # Create Chroma vector store
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    # Persist the database
    db.persist()

    return db


if __name__ == "__main__":
    # Path to the JSON file with extracted data
    json_file = "extracted_medical_data.json"

    # Create documents
    print("Creating documents from extracted data...")
    documents = create_documents_from_json(json_file)
    print(f"Created {len(documents)} documents")

    # Create vector database
    print("Creating vector database...")
    db = create_vector_db(documents)
    print(f"Vector database created and persisted to ./medical_vectordb")