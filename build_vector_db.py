
import os
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Load all PDF files
data_dir = "./history_physical_pdfs"
pdf_filenames = [f for f in os.listdir(data_dir) if f.startswith("history_physical_") and f.endswith(".pdf")]

all_docs = []
for filename in pdf_filenames:
    file_path = os.path.join(data_dir, filename)
    loader = UnstructuredPDFLoader(file_path=file_path)
    docs = loader.load()
    all_docs.extend(docs)
    print(f"Loaded {filename}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
chunks = text_splitter.split_documents(all_docs)
print(f"Total chunks created: {len(chunks)}")

# Use HuggingFace Embeddings
embedding_function = HuggingFaceEmbeddings(
    model_name="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb"
)

# Save to persistent vector database
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_function,
    persist_directory="./chroma_db",
    collection_name="simple-rag",
)
vector_db.persist()
print("Vector DB built and persisted.")
