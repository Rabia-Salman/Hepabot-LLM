import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_ollama import OllamaEmbeddings


def create_vector_db(documents: List[Document],
                     persist_directory: str = "./db/vector_db",
                     collection_name: str = "docs-hepabot-rag",
                     use_fast_embeddings: bool = True) -> Chroma:
    """
    Create or update a vector database from documents.

    Args:
        documents: List of documents to add to the database
        persist_directory: Directory to persist the database
        collection_name: Name of the collection in the database
        use_fast_embeddings: Whether to use FastEmbedEmbeddings (True) or Ollama (False)

    Returns:
        Chroma vector database instance
    """
    # Create directory if it doesn't exist
    os.makedirs(persist_directory, exist_ok=True)

    # Choose embedding model
    if use_fast_embeddings:
        embedding_model = FastEmbedEmbeddings()
        print("Using FastEmbedEmbeddings for document embedding")
    else:
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")
        print("Using Ollama nomic-embed-text for document embedding")

    # Create and persist the vector store
    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )

    # Ensure the database is persisted
    vector_db.persist()
    print(f"Vector database created with {len(documents)} documents and persisted to {persist_directory}")

    return vector_db


def load_vector_db(persist_directory: str = "./db/vector_db",
                   collection_name: str = "docs-hepabot-rag",
                   use_fast_embeddings: bool = True) -> Optional[Chroma]:
    """
    Load an existing vector database.

    Args:
        persist_directory: Directory where the database is persisted
        collection_name: Name of the collection in the database
        use_fast_embeddings: Whether to use FastEmbedEmbeddings (True) or Ollama (False)

    Returns:
        Chroma vector database instance or None if it doesn't exist
    """
    # Check if the directory exists
    if not os.path.exists(persist_directory):
        print(f"Vector database directory {persist_directory} does not exist")
        return None

    # Choose embedding model
    if use_fast_embeddings:
        embedding_model = FastEmbedEmbeddings()
    else:
        embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    try:
        # Load the existing vector store
        vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model,
            collection_name=collection_name,
        )
        print(f"Loaded vector database from {persist_directory} with {vector_db._collection.count()} documents")
        return vector_db
    except Exception as e:
        print(f"Error loading vector database: {str(e)}")
        return None