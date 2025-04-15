# app.py

import streamlit as st
from pdfragclean import ingest_pdf, split_documents, create_vector_db, create_retriever, create_chain
from langchain_ollama import ChatOllama

# Constants
MODEL_NAME = "llama3.2"

st.set_page_config(page_title="Hepabot-LLM Project", layout="centered")
def centered_header(text, level=1):
    st.markdown(f"<h{level} style='text-align: center;'>{text}</h{level}>", unsafe_allow_html=True)

centered_header(" 🏥  HEPABOT", level=2)
# st.title("     🏥  HEPABOT")
st.write("##")
st.markdown("📋 Upload a conversation, and ask questions about its content.")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Session state to store chain
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if uploaded_file is not None:
    with st.spinner("Please wait...."):
        # Save to disk
        temp_path = f"./data/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run the RAG pipeline
        data = ingest_pdf(temp_path)
        chunks = split_documents(data)
        vector_db = create_vector_db(chunks)

        # Init LLM
        llm = ChatOllama(model=MODEL_NAME)

        # Retriever and chain
        retriever = create_retriever(vector_db, llm)
        chain = create_chain(retriever, llm)

        # Store in session
        st.session_state.rag_chain = chain

    st.success("Thankyou for your patience. You can now ask questions!")

# Ask questions
if st.session_state.rag_chain:
    question = st.text_input("Ask a question about the document:")
    if question:
        with st.spinner(".........."):
            response = st.session_state.rag_chain.invoke(input=question)
        st.markdown("### 🧑‍⚕️ Answer:")
        st.write(response)
