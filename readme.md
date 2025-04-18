# Medical Records Semantic Search Application

This application allows you to perform semantic searches across medical patient records using vector embeddings. The system extracts structured data from PDF medical records, creates a vector database, and provides a user-friendly Streamlit interface for searching.

## Features

- Extract structured medical data from PDFs using LLM
- Store extracted data in a Chroma vector database for semantic search
- User-friendly Streamlit interface with multiple views:
  - Semantic search across all patient records
  - Patient record browser
  - Analytics dashboard with visualizations
- AI-enhanced query capabilities
- Filtering by demographics and document sections
- AI-generated summaries of search results

## Setup Instructions

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Ensure you have Ollama running locally with the llama3.2 model:

```bash
ollama pull llama3.2
```

3. Process your PDF files and create the vector database:

```bash
# First, extract structured data from PDFs
python enhanced_extraction.py

# Then, create the vector database
python create_vector_db.py
```

4. Run the Streamlit application:

```bash
streamlit run app.py
```

## Project Structure

- `enhanced_extraction.py` - Script to extract structured data from PDF files
- `create_vector_db.py` - Script to create a Chroma vector database from extracted data
- `app.py` - Streamlit application for searching and browsing medical records
- `utils.py` - Utility functions for PDF processing (from your original code)
- `requirements.txt` - Dependencies required for the project

## Usage

1. **Semantic Search**: Enter natural language queries to find relevant patient records
2. **Patient Browser**: Browse all patient records and view individual patient details
3. **Analytics**: View visualizations and statistics about the patient population

## Customization

- Modify the extraction template in `enhanced_extraction.py` to extract additional fields
- Add custom filters in the Streamlit app by modifying `app.py`
- Extend the analytics dashboard with additional visualizations

## Notes

- This application uses Ollama for embeddings and LLM capabilities
- The vector database is persisted to disk and can be reused across sessions
- Make sure all your PDF files follow a consistent format for best results