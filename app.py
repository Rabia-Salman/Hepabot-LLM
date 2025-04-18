# app.py
import streamlit as st
import json
import pandas as pd
import os
import tempfile
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from utils import load_conversations_from_pdf
from enhanced_extraction import extraction_template, llm, extract_metadata

# Set page configuration
st.set_page_config(
    page_title="Medical Records Semantic Search",
    page_icon="🏥",
    layout="wide"
)


# Load metadata of all patient records
@st.cache_data
def load_metadata():
    with open("extracted_medical_data.json", "r") as f:
        data = json.load(f)

    metadata_list = []
    for entry in data:
        meta = entry['metadata']
        meta['source_file'] = entry['source_file']
        metadata_list.append(meta)

    return metadata_list


# Function to load the vector database
@st.cache_resource
def load_vector_db():
    embeddings = OllamaEmbeddings(model="llama3.2")
    vectordb = Chroma(
        persist_directory="./medical_vectordb",
        embedding_function=embeddings
    )
    return vectordb


# Initialize LLM for advanced processing
@st.cache_resource
def get_llm():
    return ChatOllama(model="llama3.2", temperature=0.1)


# Setup for medical query enhancement
query_enhancement_template = ChatPromptTemplate.from_template("""
You are a medical search assistant helping to enhance user queries for vector search in a medical database.
Original query: {query}

Please enhance this query by adding relevant medical terminology, potential symptoms, or diagnostic terms 
that would help find related medical cases. Keep the enhanced query under 100 words.

Enhanced query:
""")


# Function to enhance medical queries
def enhance_query(query):
    llm = get_llm()
    chain = query_enhancement_template | llm | StrOutputParser()
    enhanced = chain.invoke({"query": query})
    return enhanced


# Function to format search results
def format_search_results(results):
    formatted_results = []

    for doc in results:
        content = doc.page_content
        metadata = doc.metadata

        formatted_results.append({
            "source": metadata.get("source", "Unknown"),
            "section": metadata.get("section", "Unknown"),
            "gender": metadata.get("gender", "Unknown"),
            "age": metadata.get("age", "Unknown"),
            "mrn": metadata.get("mrn", "Unknown"),
            "diagnosis": metadata.get("diagnosis", "Unknown"),
            "content": content
        })

    return formatted_results


# Create a chat summary from documents
def create_clinical_summary(docs):
    if not docs:
        return "Not enough relevant documents found to create a summary."

    llm = get_llm()

    # Combine the content of the top documents
    combined_content = "\n\n".join([doc.page_content for doc in docs[:5]])

    summary_template = ChatPromptTemplate.from_template("""
    You are a medical professional tasked with creating a concise clinical summary from multiple document fragments.

    Based on the following medical document excerpts, create a 2-3 paragraph summary highlighting:
    1. Common symptoms and presentations
    2. Diagnoses and conditions
    3. Treatment approaches and outcomes

    Document excerpts:
    {content}

    Clinical Summary:
    """)

    chain = summary_template | llm | StrOutputParser()
    summary = chain.invoke({"content": combined_content})
    return summary


# Main function for the app
def main():
    st.title("🏥 Medical Records Semantic Search")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "search"

    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        if st.button("Semantic Search", use_container_width=True):
            st.session_state.page = "search"
        if st.button("Patient Browser", use_container_width=True):
            st.session_state.page = "browser"
        if st.button("Analytics", use_container_width=True):
            st.session_state.page = "analytics"
        if st.button("Generate Report", use_container_width=True):
            st.session_state.page = "generate_report"

        st.divider()
        st.write(
            "This application provides semantic search capabilities for medical records. It uses vector embeddings to find relevant patient records based on your query.")

    # Page selection
    if st.session_state.page == "search":
        show_search_page()
    elif st.session_state.page == "browser":
        show_browser_page()
    elif st.session_state.page == "analytics":
        show_analytics_page()
    elif st.session_state.page == "generate_report":
        show_generate_report_page()


def show_search_page():
    st.header("Semantic Search")
    st.write("Search across all patient records using natural language queries")

    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Enter your search query:",
                                     placeholder="E.g., Patients with kidney stones and HCV")

    with col2:
        use_ai = st.checkbox("Enhance query with AI", value=True)
        search_btn = st.button("Search", use_container_width=True)

    # Filter options
    with st.expander("Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender_filter = st.selectbox("Gender", ["Any", "Male", "Female"])

        with col2:
            age_range = st.slider("Age Range", 0, 100, (0, 100))

        with col3:
            sections = ["Any", "Active Symptoms", "Diagnostic Conclusions", "Therapeutic Interventions",
                        "Diagnostic Evidence", "Chronic Conditions", "Follow-up Plan", "Summary Narrative"]
            section_filter = st.selectbox("Document Section", sections)

    if search_query and search_btn:
        # Load vector database
        vectordb = load_vector_db()

        # Enhance query if selected
        if use_ai:
            with st.spinner("Enhancing query with medical terminology..."):
                enhanced_query = enhance_query(search_query)
                st.info(f"Enhanced query: {enhanced_query}")
                query_for_search = enhanced_query
        else:
            query_for_search = search_query

        # Prepare metadata filters
        metadata_filters = {}
        if gender_filter != "Any":
            metadata_filters["gender"] = gender_filter
        if section_filter != "Any":
            metadata_filters["section"] = section_filter

        # Search the database
        with st.spinner("Searching for relevant records..."):
            results = vectordb.similarity_search(
                query_for_search,
                k=10,
                filter=metadata_filters if metadata_filters else None
            )

            formatted_results = format_search_results(results)

        # Display results
        if formatted_results:
            # Generate AI summary
            with st.spinner("Generating clinical summary..."):
                summary = create_clinical_summary(results)
                st.subheader("AI-Generated Summary")
                st.write(summary)

            # Show individual results
            st.subheader(f"Search Results ({len(formatted_results)} found)")

            for i, result in enumerate(formatted_results):
                with st.expander(f"Patient: {result['mrn']} - {result['diagnosis']} ({result['section']})"):
                    st.write(f"**Source:** {result['source']}")
                    st.write(f"**Demographics:** {result['gender']}, {result['age']} years old")
                    st.write(f"**Section:** {result['section']}")
                    st.divider()
                    st.write(result['content'])
        else:
            st.warning("No results found. Try modifying your search query.")


def show_browser_page():
    st.header("Patient Record Browser")

    # Load metadata
    metadata_list = load_metadata()

    # Create dataframe
    df = pd.DataFrame(metadata_list)

    # Display table with filters
    st.dataframe(df, use_container_width=True)

    # Patient detail view
    st.subheader("Patient Detail View")

    # Select patient by MRN
    selected_mrn = st.selectbox("Select Patient MRN", sorted(df['mrn'].unique()))

    if selected_mrn:
        patient_data = df[df['mrn'] == selected_mrn].iloc[0]

        # Display patient info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("MRN", patient_data['mrn'])
        with col2:
            st.metric("Age", patient_data['age'])
        with col3:
            st.metric("Gender", patient_data['gender'])

        st.subheader(f"Diagnosis: {patient_data['diagnosis']}")

        # Load full patient data
        with open("extracted_medical_data.json", "r") as f:
            data = json.load(f)

        # Find the patient record
        patient_record = next((item for item in data if item['metadata']['mrn'] == selected_mrn), None)

        if patient_record:
            st.text_area("Full Clinical Summary", patient_record['raw_text'], height=400)


def show_analytics_page():
    st.header("Medical Records Analytics")

    # Load metadata
    metadata_list = load_metadata()
    df = pd.DataFrame(metadata_list)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Demographics")

        # Gender distribution
        gender_counts = df['gender'].value_counts()
        st.bar_chart(gender_counts)

        # Age distribution
        st.subheader("Age Distribution")
        age_chart = pd.DataFrame(df['age'].astype(int))
        # Fixed histogram chart
        hist_values = pd.DataFrame(df['age'].astype(int).value_counts().sort_index())
        st.bar_chart(hist_values)

    with col2:
        st.subheader("Diagnosis Distribution")

        # Count diagnoses
        diagnosis_counts = df['diagnosis'].value_counts().head(10)
        st.bar_chart(diagnosis_counts)

        # Average age by diagnosis
        # st.subheader("Average Age by Top Diagnoses")
        # avg_age = df.groupby('diagnosis')['age'].mean().sort_values(ascending=False).head(5)
        # st.bar_chart(avg_age)


def show_generate_report_page():
    st.header("🩺 Generate Patient Report")
    st.write("Upload a clinical conversation PDF to generate a structured medical report")

    uploaded_file = st.file_uploader("Upload a clinical conversation PDF", type=["pdf"])

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        with st.spinner("🔍 Analyzing data ..."):
            try:
                # Extract metadata first
                metadata = extract_metadata(tmp_path)

                # Then load conversation
                turns = load_conversations_from_pdf(tmp_path)
                conversation_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])

                chain = extraction_template | llm | StrOutputParser()
                # Add metadata to the invocation
                raw_output = chain.invoke({
                    "gender": metadata["gender"],
                    "age": metadata["age"],
                    "mrn": metadata["mrn"],
                    "diagnosis": metadata["diagnosis"],
                    "conversation": conversation_text
                })

                # Clean and save output in session state
                def clean_output(text):
                    cleaned = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
                    return cleaned.strip()

                cleaned_output = clean_output(raw_output)
                st.session_state["raw_output"] = raw_output
                st.session_state["cleaned_output"] = cleaned_output

                # Display extracted metadata
                metadata_col1, metadata_col2 = st.columns(2)
                with metadata_col1:
                    st.subheader("Patient Information")
                    st.write(f"**Gender:** {metadata['gender']}")
                    st.write(f"**Age:** {metadata['age']}")

                with metadata_col2:
                    st.write(f"**MRN:** {metadata['mrn']}")
                    st.write(f"**Diagnosis:** {metadata['diagnosis']}")

            except Exception as e:
                st.error(f"❌ Error during extraction: {e}")

    # This will render after the initial analysis is done
    if "raw_output" in st.session_state:
        st.subheader("📄 Medical Report")
        with st.expander("🔍 Show Details", expanded=True):
            st.text_area("Summary Report", st.session_state.raw_output, height=400)

        if st.button("📥 Save Report as PDF"):
            pdf_path = Path(tempfile.gettempdir()) / "medical_summary.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            width, height = letter

            # Add a title to the PDF
            c.setFont("Helvetica-Bold", 16)
            c.drawString(40, height - 40, "Medical Report Summary")
            c.setFont("Helvetica", 12)

            # Draw a line under the title
            c.line(40, height - 50, width - 40, height - 50)

            # Start content below the line
            y = height - 70

            lines = st.session_state.raw_output.splitlines()
            max_chars_per_line = 100
            formatted_lines = []
            for line in lines:
                if len(line) <= max_chars_per_line:
                    formatted_lines.append(line)
                else:
                    while len(line) > max_chars_per_line:
                        formatted_lines.append(line[:max_chars_per_line])
                        line = line[max_chars_per_line:]
                    if line:
                        formatted_lines.append(line)

            for line in formatted_lines:
                if line.strip() and any(heading in line for heading in
                                        ["Patient Demographics:", "Active Symptoms:", "Negative Findings:",
                                         "Diagnostic Conclusions:", "Therapeutic Interventions:",
                                         "Diagnostic Evidence:", "Chronic Conditions:", "Follow-up Plan:",
                                         "Visit Timeline:", "Summary Narrative:"]):
                    # This is a heading, make it bold
                    c.setFont("Helvetica-Bold", 12)
                    c.drawString(40, y, line)
                    y -= 20  # Extra space after heading
                    c.setFont("Helvetica", 12)
                else:
                    c.drawString(40, y, line)
                    y -= 14

                if y < 40:
                    c.showPage()
                    y = height - 40

            c.save()

            with open(pdf_path, "rb") as f:
                st.download_button("Download PDF", f, file_name="medical_summary.pdf")

            st.success("PDF report generated successfully!")

            # Preview the PDF
            st.subheader("PDF Preview")
            try:
                pdf_url = pdf_path.as_uri()
                st.components.v1.iframe(pdf_url, width=700, height=500)
            except:
                st.warning("PDF preview not available. You can download the file using the button above.")


if __name__ == "__main__":
    main()