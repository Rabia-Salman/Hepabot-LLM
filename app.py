import streamlit as st
import json
import pandas as pd
import os
import tempfile
from pathlib import Path
from enhanced_extraction import extract_metadata, load_conversations_from_pdf, process_pdf
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import time
import base64
from dotenv import load_dotenv
from io import BytesIO
from typing import List
import requests
from audio_transcription import process_audio_to_pdf
import altair as alt

# Import vector_db functions from the first file
from vector_db import initialize_vector_db, add_patient_record, search_records
from extract_each_patient_json import save_patient_records

# Import utility modules from the second file
from utils import process_documents, create_vector_db, load_vector_db, create_rag_chain, ask_question

# Optional voice support
try:
    from elevenlabs import Voice, VoiceSettings, generate, play
    from elevenlabs.api import User
    ELEVENLABS_AVAILABLE = True
except ImportError as e:
    print(f"ImportError: {e}")
    ELEVENLABS_AVAILABLE = False

# Constants from the second file
DB_PATH = "./db/vector_db"
COLLECTION_NAME = "docs-hepabot-rag"
MODEL_NAME = "llama3.2"
ALLOWED_EXTENSIONS = ['.pdf', '.txt', '.json']
input_json_path = "doctor_patient_data_80.json"
output_directory = "split_patient_files"

# Set page configuration
st.set_page_config(
    page_title="HEPABOT",
    page_icon="🏥",
    layout="wide"
)


# Load metadata of all patient records
@st.cache_data
def load_metadata(_refresh=False):
    try:
        with open(input_json_path, "r") as f:
            data = json.load(f)

        metadata_list = []
        for entry in data:
            if 'structured_data' in entry and 'patient_id' in entry:
                meta = entry['structured_data'].get('PatientDemographics', {})
                meta['patient_id'] = entry['patient_id']
                metadata_list.append(meta)
            else:
                print(f"Skipping entry missing required fields: {entry}")

        return metadata_list
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return []


# Function to load and initialize the vector database
@st.cache_resource
def get_vector_db():
    collection = initialize_vector_db()
    return collection


# Function to format search results for display
def format_search_results(results):
    formatted_results = []

    if results and 'documents' in results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if 'metadatas' in results else {}

            formatted_results.append({
                "patient_id": metadata.get("patient_id", "Unknown"),
                "gender": metadata.get("gender", "Unknown"),
                "age": metadata.get("age", "Unknown"),
                "mrn": metadata.get("mrn", "Unknown"),
                "diagnosis": metadata.get("diagnosis", "Unknown"),
                "content": doc
            })

    return formatted_results


# Main function for the app
def main():
    st.title("🏥 HEPABOT")

    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "search"
    if 'refresh_data' not in st.session_state:
        st.session_state.refresh_data = False

    # Sidebar for navigation
    with st.sidebar:
        st.title("Navigation")
        if st.button("Advance Search", use_container_width=True):
            st.session_state.page = "search"
        if st.button("Patient Browser", use_container_width=True):
            st.session_state.page = "browser"
        if st.button("Analytics", use_container_width=True):
            st.session_state.page = "analytics"
        if st.button("Generate Report", use_container_width=True):
            st.session_state.page = "generate_report"
        if st.button("Clinical Assistant", use_container_width=True):
            st.session_state.page = "clinical_assistant"
        # if st.button("Disease Diagnosis", use_container_width=True):
        #     st.session_state.page = "disease_diagnosis"

    # Page selection
    if st.session_state.page == "search":
        show_search_page()
    elif st.session_state.page == "browser":
        show_browser_page()
    elif st.session_state.page == "analytics":
        show_analytics_page()
    elif st.session_state.page == "generate_report":
        show_generate_report_page()
    elif st.session_state.page == "clinical_assistant":
        show_clinical_assistant_page()
    # elif st.session_state.page == "disease_diagnosis":
    #     show_disease_diagnosis_page()



def save_uploaded_files(uploaded_files) -> List[str]:
    """Save uploaded files and return their paths"""
    temp_dir = tempfile.mkdtemp()
    file_paths = []

    for uploaded_file in uploaded_files:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension not in ALLOWED_EXTENSIONS:
            st.warning(f"Unsupported file format: {file_extension}. Skipping {uploaded_file.name}")
            continue

        temp_file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(temp_file_path)

    return file_paths

def text_to_speech(text: str, api_key: str = None) -> BytesIO:
    """Convert text to speech using ElevenLabs API"""
    if not ELEVENLABS_AVAILABLE:
        st.error("ElevenLabs package is not installed. Voice over is not available.")
        return None

    if not api_key:
        api_key = os.getenv("ELEVENLABS_API_KEY")

    if not api_key:
        st.error("ElevenLabs API key is not set. Voice over is not available.")
        return None

    try:
        import elevenlabs
        elevenlabs.set_api_key(api_key)

        voices = elevenlabs.voices()
        if not voices:
            st.error("No voices available in your ElevenLabs account")
            return None
        voice_id = voices[0].voice_id

        audio = elevenlabs.generate(
            text=text,
            voice=st.session_state.get("selected_voice_id", voice_id),
            model="eleven_turbo_v2"
        )

        return BytesIO(audio)
    except Exception as e:
        st.error(f"Error generating voice: {str(e)}")
        return None

def get_download_link(data, filename, text):
    """Generate a download link for a file"""
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{text}</a>'
    return href


def show_clinical_assistant_page():
    st.header("🩺 Clinical Assistant")
    st.write("Upload medical documents and ask questions to get AI-assisted medical insights.")

    # Initialize session state variables
    if 'vector_db_created' not in st.session_state:
        st.session_state.vector_db_created = False
    if 'rag_chain' not in st.session_state:
        st.session_state.rag_chain = None
    if 'last_response' not in st.session_state:
        st.session_state.last_response = ""
    if 'selected_voice_id' not in st.session_state:
        st.session_state.selected_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Default voice ID (Adam)

    # Check if database exists at startup
    if os.path.exists(DB_PATH) and not st.session_state.vector_db_created:
        st.session_state.vector_db_created = True

    # Sidebar for database operations
    with st.sidebar:
        st.header("Document Management")

        # Check if database exists
        db_exists = os.path.exists(DB_PATH)
        if db_exists:
            st.success("Vector database exists! Ready to answer questions.")
        else:
            st.warning("No vector database found. Please upload documents.")

        # Upload files section
        uploaded_files = st.file_uploader(
            "Upload PDF, TXT, or JSON files",
            accept_multiple_files=True,
            type=["pdf", "txt", "json"]
        )

        # Database creation options
        with st.expander("Advanced Options", expanded=False):
            chunk_size = st.number_input("Chunk Size", value=1200, min_value=500, max_value=2000)
            chunk_overlap = st.number_input("Chunk Overlap", value=300, min_value=0, max_value=500)
            # use_fast_embeddings = st.checkbox("Use Fast Embeddings", value=True)

        # Create database button
        if st.button("Process Documents & Create Database"):
            if not uploaded_files:
                st.error("Please upload at least one document.")
            else:
                with st.status("Processing documents..."):
                    # Save the uploaded files to disk
                    file_paths = save_uploaded_files(uploaded_files)

                    if file_paths:
                        # Process the documents
                        st.text(f"Processing {len(file_paths)} documents...")
                        docs = process_documents(
                            file_paths,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )

                        # Create or update the vector database
                        st.text("Creating vector database...")
                        vector_db = create_vector_db(
                            docs,
                            persist_directory=DB_PATH,
                            collection_name=COLLECTION_NAME
                        )

                        st.session_state.vector_db_created = True
                        st.success(f"Vector database created with {len(docs)} chunks!")
                    else:
                        st.error("No valid documents were uploaded.")

        # Delete database button
        if st.button("Delete Database"):
            if os.path.exists(DB_PATH):
                import shutil
                shutil.rmtree(DB_PATH)
                st.session_state.vector_db_created = False
                st.session_state.rag_chain = None
                st.success("Database deleted successfully.")
            else:
                st.info("No database to delete.")

    # Main area for question answering
    st.subheader("Medical Diagnosis Assistant")

    # Check if chain is loaded or needs to be loaded
    if st.session_state.vector_db_created and not st.session_state.rag_chain:
        with st.status("Loading RAG chain..."):
            try:
                # Load the vector database
                vector_db = load_vector_db(
                    persist_directory=DB_PATH,
                    collection_name=COLLECTION_NAME
                )

                if vector_db:
                    # Create RAG chain
                    st.session_state.rag_chain = create_rag_chain(vector_db, MODEL_NAME)
                    st.success("Ready to answer your medical questions!")
                else:
                    st.error("Failed to load vector database. Please create a new one.")
            except Exception as e:
                st.error(f"Error loading database: {str(e)}")
                st.session_state.vector_db_created = False

    # Question input
    if st.session_state.vector_db_created:
        col1, col2 = st.columns([3, 1])

        with col1:
            question = st.text_area(
                "Enter your medical question:",
                height=100,
                placeholder="Example: Patient has age 70 and shows symptoms of nausea and abdominal pain with fatigue, find disease"
            )

        with col2:
            voice_enabled = st.checkbox("Enable voice output", value=ELEVENLABS_AVAILABLE)

            if voice_enabled:
                if not ELEVENLABS_AVAILABLE:
                    st.warning("ElevenLabs package is not installed. Voice output disabled.")
                    voice_enabled = False
                else:
                    api_key = None
                    if not os.getenv("ELEVENLABS_API_KEY"):
                        api_key = st.text_input("ElevenLabs API Key", type="password")
                        if api_key:
                            os.environ["ELEVENLABS_API_KEY"] = api_key

                    # Try to show available voices if API key is provided
                    if api_key or os.getenv("ELEVENLABS_API_KEY"):
                        try:
                            import elevenlabs
                            elevenlabs.set_api_key(api_key or os.getenv("ELEVENLABS_API_KEY"))
                            voices = elevenlabs.voices()
                            if voices:
                                voice_options = {voice.name: voice.voice_id for voice in voices}
                                selected_voice = st.selectbox("Select voice", options=list(voice_options.keys()))
                                st.session_state.selected_voice_id = voice_options[selected_voice]
                            else:
                                st.info("No custom voices found. Will use default voice.")
                        except Exception:
                            st.info("Could not fetch voices. Will use default voice.")

        # Submit button
        if st.button("Get Diagnosis"):
            if not question:
                st.warning("Please enter a question.")
            elif not st.session_state.rag_chain:
                st.error("RAG chain is not loaded. Please create or load a database first.")
            else:
                with st.status("Generating answer..."):
                    try:
                        # Get answer from RAG chain
                        response = ask_question(st.session_state.rag_chain, question)
                        st.session_state.last_response = response
                    except Exception as e:
                        st.error(f"Error generating diagnosis: {str(e)}")

        # Display the response
        if st.session_state.last_response:
            st.subheader("Diagnosis Result:")
            st.markdown(st.session_state.last_response)

            col1, col2 = st.columns(2)

            # Download button
            with col1:
                if st.button("Download Result"):
                    download_data = st.session_state.last_response.encode()
                    st.markdown(
                        get_download_link(
                            download_data,
                            "diagnosis_result.txt",
                            "Download Diagnosis Result"
                        ),
                        unsafe_allow_html=True
                    )

            # Voice playback
            with col2:
                if voice_enabled and st.button("Play Voice"):
                    with st.spinner("Generating voice..."):
                        audio_data = text_to_speech(st.session_state.last_response, api_key)
                        if audio_data:
                            st.audio(audio_data, format='audio/mp3')
    else:
        st.info("Please upload documents and create a vector database to start asking questions.")

def show_disease_diagnosis_page():

    # Define diagnostic criteria for liver diseases
    liver_diseases = {
        "Alcoholic Liver Disease": {
            "symptoms": ["jaundice", "abdominal pain", "fatigue"],
            "history": ["alcohol use", "heavy drinking"],
            "labs": [
                {"test": "ALT", "condition": "elevated", "threshold": 45},
                {"test": "AST", "condition": "elevated", "threshold": 35},
                {"test": "albumin", "condition": "low", "threshold": 3.5},
                {"type": "ratio", "tests": ["AST", "ALT"], "condition": "AST > ALT"}
            ],
            "age_factor": {"min": 30, "max": 60, "boost": 10}
        },
        "Viral Hepatitis": {
            "symptoms": ["jaundice", "fatigue", "nausea"],
            "history": ["viral infection", "hepatitis", "IV drug use", "unprotected sex"],
            "labs": [
                {"test": "ALT", "condition": "elevated", "threshold": 45},
                {"test": "AST", "condition": "elevated", "threshold": 35}
            ],
            "age_factor": {"min": 0, "max": 100, "boost": 5}
        },
        "Cirrhosis": {
            "symptoms": ["jaundice", "ascites", "fatigue"],
            "history": ["alcohol use", "hepatitis", "chronic liver disease"],
            "labs": [
                {"test": "albumin", "condition": "low", "threshold": 3.5},
                {"test": "bilirubin", "condition": "elevated", "threshold": 1.2}
            ],
            "imaging": ["nodular liver"],
            "age_factor": {"min": 40, "max": 80, "boost": 10}
        },
        "Fatty Liver Disease": {
            "symptoms": ["fatigue", "abdominal discomfort"],
            "history": ["obesity", "diabetes", "metabolic syndrome"],
            "labs": [
                {"test": "ALT", "condition": "elevated", "threshold": 45},
                {"test": "AST", "condition": "elevated", "threshold": 35}
            ],
            "imaging": ["fatty liver"],
            "age_factor": {"min": 30, "max": 70, "boost": 5}
        },
        "Liver Cancer": {
            "symptoms": ["abdominal pain", "weight loss"],
            "labs": [
                {"test": "AFP", "condition": "elevated", "threshold": 20}
            ],
            "imaging": ["mass", "tumor"],
            "age_factor": {"min": 50, "max": 100, "boost": 15}
        }
    }

    # Define lab tests and their units
    lab_tests = {
        "ALT": "U/L",
        "AST": "U/L",
        "bilirubin": "mg/dL",
        "albumin": "g/dL",
        "AFP": "ng/mL"
    }

    # Extraction functions
    def extract_age(text):
        age_match = re.search(r'(\d+)-year-old', text)
        if age_match:
            return int(age_match.group(1))
        age_match = re.search(r'age (\d+)', text)
        if age_match:
            return int(age_match.group(1))
        return None

    def extract_symptoms(text):
        all_symptoms = set()
        for disease in liver_diseases.values():
            all_symptoms.update(disease.get("symptoms", []))
        return [s for s in all_symptoms if s in text.lower()]

    def extract_history(text):
        return text.lower()

    def extract_labs(text):
        labs = {}
        for test, unit in lab_tests.items():
            pattern = rf"{test}\s*(\d+\.?\d*)\s*{unit}"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                labs[test] = float(match.group(1))
        return labs

    def extract_imaging(text):
        all_imaging = set()
        for disease in liver_diseases.values():
            all_imaging.update(disease.get("imaging", []))
        return [i for i in all_imaging if i in text.lower()]

    def format_labs(labs):
        if not labs:
            return "None detected"
        return ", ".join([f"{test} {value} {lab_tests[test]}" for test, value in labs.items()])

    def calculate_score(extracted, disease):
        score = 0
        matching_criteria = []

        # Check symptoms
        if "symptoms" in disease:
            matching_symptoms = [s for s in disease["symptoms"] if s in extracted["symptoms"]]
            score += len(matching_symptoms) * 10
            if matching_symptoms:
                matching_criteria.append(f"{len(matching_symptoms)} symptom(s)")

        # Check history
        if "history" in disease:
            matching_history = [h for h in disease["history"] if h in extracted["history"]]
            score += len(matching_history) * 15
            if matching_history:
                matching_criteria.append(f"{len(matching_history)} history factor(s)")

        # Check labs
        if "labs" in disease:
            for lab_cond in disease["labs"]:
                if "type" in lab_cond and lab_cond["type"] == "ratio":
                    if lab_cond["condition"] == "AST > ALT" and extracted["labs"].get("AST", 0) > extracted["labs"].get(
                            "ALT", 0):
                        score += 20
                        matching_criteria.append("AST > ALT")
                else:
                    test = lab_cond["test"]
                    if test in extracted["labs"]:
                        value = extracted["labs"][test]
                        if lab_cond["condition"] == "elevated" and value > lab_cond["threshold"]:
                            score += 20
                            matching_criteria.append(f"{test} elevated")
                        elif lab_cond["condition"] == "low" and value < lab_cond["threshold"]:
                            score += 20
                            matching_criteria.append(f"{test} low")

        # Check imaging
        if "imaging" in disease:
            matching_imaging = [i for i in disease["imaging"] if i in extracted["imaging"]]
            score += len(matching_imaging) * 20
            if matching_imaging:
                matching_criteria.append(f"{len(matching_imaging)} imaging finding(s)")

        # Check age
        if extracted["age"] and "age_factor" in disease:
            if disease["age_factor"]["min"] <= extracted["age"] <= disease["age_factor"]["max"]:
                score += disease["age_factor"]["boost"]
                matching_criteria.append("age within range")

        return score, matching_criteria

    # Streamlit app for disease diagnosis page
    def show_disease_diagnosis_page():
        st.header("Liver Disease Diagnosis Assistant")
        st.write("Enter a patient description or medical notes below to analyze possible liver disease diagnoses.")
        input_text = st.text_area("Patient Description",
                                  placeholder="E.g., A 45-year-old male with a history of heavy alcohol use presents with jaundice, abdominal pain, and fatigue. Lab tests show ALT 150 U/L, AST 200 U/L, bilirubin 3.5 mg/dL, albumin 3.0 g/dL.")

        if st.button("Analyze"):
            # Extract data
            extracted = {
                "age": extract_age(input_text),
                "symptoms": extract_symptoms(input_text),
                "history": extract_history(input_text),
                "labs": extract_labs(input_text),
                "imaging": extract_imaging(input_text)
            }

            # Calculate diagnoses
            diagnoses = []
            for disease_name, criteria in liver_diseases.items():
                score, matching_criteria = calculate_score(extracted, criteria)
                if score > 0:
                    diagnoses.append({
                        "name": disease_name,
                        "score": score,
                        "matching_criteria": matching_criteria
                    })

            # Sort diagnoses by score descending
            diagnoses.sort(key=lambda x: x["score"], reverse=True)

            # Display results
            st.subheader("Extracted Information")
            st.write(f"**Age:** {extracted['age'] or 'Not specified'}")
            st.write(f"**Symptoms:** {', '.join(extracted['symptoms']) if extracted['symptoms'] else 'None detected'}")
            st.write(f"**Medical History:** {extracted['history'] if extracted['history'] else 'None detected'}")
            st.write(f"**Lab Tests:** {format_labs(extracted['labs'])}")
            st.write(f"**Imaging:** {', '.join(extracted['imaging']) if extracted['imaging'] else 'Not specified'}")

            st.subheader("Suggested Diagnoses")
            if diagnoses:
                for d in diagnoses:
                    st.write(f"**{d['name']}** - Confidence Score: {d['score']}")
                    st.write(f"Matching Criteria: {', '.join(d['matching_criteria'])}")
            else:
                st.write("No clear diagnosis could be determined from the input.")

            st.write(
                "**Disclaimer:** This tool is for informational purposes only and not a substitute for professional medical advice. Consult a healthcare professional for an accurate diagnosis.")

    # Call this function in your app.py when the disease diagnosis page is selected
    show_disease_diagnosis_page()

def show_search_page():
    st.header("Diagnosis Search")
    st.write("Search for patients with specific diagnoses or conditions")

    # Clear cache if refresh is needed
    if st.session_state.refresh_data:
        st.cache_data.clear()
        st.session_state.refresh_data = False

    # Load metadata
    metadata_list = load_metadata(_refresh=st.session_state.refresh_data)

    # Get unique diagnoses from the metadata
    all_diagnoses = []
    for meta in metadata_list:
        diagnosis = meta.get('Diagnosis', '')
        if diagnosis and diagnosis not in all_diagnoses:
            all_diagnoses.append(diagnosis)

    # Sort diagnoses alphabetically
    all_diagnoses.sort()

    # Add an "All" option
    all_diagnoses = ["All"] + all_diagnoses

    # Create search interface
    col1, col2 = st.columns([3, 1])

    with col1:
        search_query = st.text_input("Search by keyword in diagnosis:",
                                     placeholder="E.g., cancer, diabetes, heart")

    with col2:
        selected_diagnosis = st.selectbox("Filter by diagnosis:", all_diagnoses)

    # Filter options
    with st.expander("Advanced Filters"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender_filter = st.selectbox("Gender", ["Any", "Male", "Female"])

        with col2:
            age_range = st.slider("Age Range", 0, 100, (0, 100))

        with col3:
            sort_by = st.selectbox("Sort by", ["Diagnosis", "Age", "MRN"])

    # Apply filters
    filtered_patients = []

    for patient in metadata_list:
        diagnosis = patient.get('Diagnosis', '')
        include = True

        # Apply diagnosis filter
        if selected_diagnosis != "All" and diagnosis != selected_diagnosis:
            include = False

        # Apply keyword search
        if search_query and search_query.lower() not in diagnosis.lower():
            include = False

        # Apply gender filter
        if gender_filter != "Any" and patient.get('Gender', '') != gender_filter:
            include = False

        # Apply age filter
        try:
            patient_age = int(patient.get('Age', 0))
            if patient_age < age_range[0] or patient_age > age_range[1]:
                include = False
        except (ValueError, TypeError):
            pass

        if include:
            filtered_patients.append(patient)

    # Sort the results
    if sort_by == "Diagnosis":
        filtered_patients.sort(key=lambda x: x.get('Diagnosis', ''))
    elif sort_by == "Age":
        # Convert age to int for sorting, with a default of 0 if conversion fails
        filtered_patients.sort(key=lambda x: int(x.get('Age', 0)) if x.get('Age', '').isdigit() else 0)
    elif sort_by == "MRN":
        filtered_patients.sort(key=lambda x: x.get('MRN', ''))

    # Display results
    st.subheader(f"Search Results ({len(filtered_patients)} patients found)")

    if filtered_patients:
        # Create a DataFrame for display
        df_display = []
        for patient in filtered_patients:
            df_display.append({
                "Patient ID": patient.get('patient_id', 'Unknown'),
                "MRN": patient.get('MRN', 'Unknown'),
                "Gender": patient.get('Gender', 'Unknown'),
                "Age": patient.get('Age', 'Unknown'),
                "Diagnosis": patient.get('Diagnosis', 'Unknown')
            })

        # Display as a table with ability to select a row
        df = pd.DataFrame(df_display)
        selected_indices = st.dataframe(df, use_container_width=True, height=300)

        # Allow selecting a patient for detailed view
        selected_patient_id = st.selectbox("Select a patient to view details:",
                                           [''] + [p.get('patient_id', '') for p in filtered_patients])

        if selected_patient_id:
            # Find the selected patient in the JSON data
            try:
                with open(input_json_path, "r") as f:
                    all_data = json.load(f)

                # Find the selected patient record
                patient_record = next((item for item in all_data if item.get('patient_id') == selected_patient_id),
                                      None)

                if patient_record:
                    # Display patient details
                    st.subheader("Patient Details")

                    # Get demographics
                    demographics = patient_record.get('structured_data', {}).get('PatientDemographics', {})

                    # Display in columns
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MRN", demographics.get('MRN', 'N/A'))
                    with col2:
                        st.metric("Age", demographics.get('Age', 'N/A'))
                    with col3:
                        st.metric("Gender", demographics.get('Gender', 'N/A'))
                    with col4:
                        st.metric("Diagnosis", demographics.get('Diagnosis', 'N/A'))

                    # Display tabs for different sections of structured data
                    tabs = st.tabs(["Summary Report", "Clinical Summary", "Diagnostic Information", "Treatment Plan",
                                    "Conversation"])
                    with tabs[0]:
                        summary_report = patient_record.get('structured_data', {})
                        st.subheader("Summary Report")
                        st.write(summary_report)

                    with tabs[1]:
                        clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})

                        st.subheader("Clinical Summary")

                        # Active symptoms
                        st.write("**Active Symptoms:**")
                        symptoms = clinical_summary.get('ActiveSymptoms', [])
                        if symptoms:
                            for symptom in symptoms:
                                st.write(f"- {symptom}")
                        else:
                            st.write("No active symptoms recorded")

                        # Negative findings
                        st.write("**Negative Findings:**")
                        neg_findings = clinical_summary.get('NegativeFindings', [])
                        if neg_findings:
                            for finding in neg_findings:
                                st.write(f"- {finding}")
                        else:
                            st.write("No negative findings recorded")

                    with tabs[2]:
                        # Display diagnostic information
                        st.subheader("Diagnostic Information")

                        # Diagnostic conclusions
                        diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions', [])
                        if diag_conclusions:
                            st.write("**Diagnostic Conclusions:**")
                            for conclusion in diag_conclusions:
                                st.write(f"- {conclusion}")

                        # Diagnostic evidence
                        diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})

                        # Imaging findings
                        img_findings = diag_evidence.get('ImagingFindings', [])
                        if img_findings:
                            st.write("**Imaging Findings:**")
                            for finding in img_findings:
                                st.write(f"- {finding}")

                        # Lab results
                        lab_results = diag_evidence.get('LabResults', [])
                        if lab_results:
                            st.write("**Laboratory Results:**")
                            for result in lab_results:
                                st.write(f"- {result}")

                        # Pathology findings
                        path_findings = diag_evidence.get('PathologyFindings', [])
                        if path_findings:
                            st.write("**Pathology Findings:**")
                            for finding in path_findings:
                                st.write(f"- {finding}")

                        # Chronic conditions
                        chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})

                        # Chronic diseases
                        diseases = chronic.get('ChronicDiseases', [])
                        if diseases:
                            st.write("**Chronic Diseases:**")
                            for disease in diseases:
                                st.write(f"- {disease}")

                        # Comorbidities
                        comorbidities = chronic.get('Comorbidities', [])
                        if comorbidities:
                            st.write("**Comorbidities:**")
                            for comorbidity in comorbidities:
                                st.write(f"- {comorbidity}")

                    with tabs[3]:
                        # Display treatment plan
                        st.subheader("Treatment and Follow-up Plan")

                        # Therapeutic interventions
                        therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})

                        # Medications
                        medications = therapies.get('Medications', [])
                        if medications:
                            st.write("**Medications:**")
                            for med in medications:
                                st.write(f"- {med}")

                        # Procedures
                        procedures = therapies.get('Procedures', [])
                        if procedures:
                            st.write("**Procedures:**")
                            for proc in procedures:
                                st.write(f"- {proc}")

                        # Follow-up plan
                        followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})

                        # Planned consultations
                        consultations = followup.get('PlannedConsultations', [])
                        if consultations:
                            st.write("**Planned Consultations:**")
                            for consult in consultations:
                                st.write(f"- {consult}")

                        # Scheduled tests
                        tests = followup.get('ScheduledTests', [])
                        if tests:
                            st.write("**Scheduled Tests:**")
                            for test in tests:
                                st.write(f"- {test}")

                        # Next appointment
                        appointments = followup.get('NextAppointmentDetails', [])
                        if appointments:
                            st.write("**Next Appointment Details:**")
                            for appt in appointments:
                                st.write(f"- {appt}")

                        # Visit timeline
                        timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
                        if timeline:
                            st.write("**Visit Timeline:**")
                            for visit in timeline:
                                st.write(f"- {visit}")

                    with tabs[4]:
                        # Display raw conversation
                        st.text_area("Raw Conversation Text",
                                     patient_record.get('raw_text', 'No conversation text available'), height=400)

                else:
                    st.warning("Patient record not found in the database")

            except Exception as e:
                st.error(f"Error retrieving patient data: {e}")
    else:
        st.warning("No patients match your search criteria. Please adjust your filters.")


def show_browser_page():
    st.header("Patient Record Browser")

    # Clear cache if refresh is needed
    if st.session_state.refresh_data:
        st.cache_data.clear()
        st.session_state.refresh_data = False

    # Load metadata with refresh
    metadata_list = load_metadata(_refresh=True)

    # Create dataframe
    if metadata_list:
        df = pd.DataFrame(metadata_list)

        # Sort options and filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_option = st.selectbox(
                "Sort by:",
                ["patient_id", "Age", "Gender", "MRN", "Diagnosis"]
            )

        with col2:
            filter_gender = st.selectbox(
                "Filter by gender:",
                ["All", "Male", "Female"]
            )

        with col3:
            search_term = st.text_input("Search by MRN or diagnosis:")

        # Apply filters
        filtered_df = df.copy()

        if filter_gender != "All":
            filtered_df = filtered_df[filtered_df["Gender"] == filter_gender]

        if search_term:
            # Search in MRN and Diagnosis columns
            search_mask = (
                    filtered_df["MRN"].astype(str).str.contains(search_term, case=False, na=False) |
                    filtered_df["Diagnosis"].astype(str).str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]

        # Sort dataframe
        if sort_option in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_option)

        # Display table with filters
        st.dataframe(filtered_df, use_container_width=True)

        # Patient detail view
        st.subheader("Patient Detail View")

        # Select patient by ID
        selected_id = st.selectbox("Select Patient ID", sorted([m['patient_id'] for m in metadata_list]))

        if selected_id:
            # Find patient in metadata
            patient_data = next((m for m in metadata_list if m['patient_id'] == selected_id), None)

            if patient_data:
                # Display patient info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MRN", patient_data.get('MRN', 'N/A'))
                with col2:
                    st.metric("Age", patient_data.get('Age', 'N/A'))
                with col3:
                    st.metric("Gender", patient_data.get('Gender', 'N/A'))

                st.subheader(f"Diagnosis: {patient_data.get('Diagnosis', 'N/A')}")

                # Load full patient data from combined JSON
                try:
                    with open(input_json_path, "r") as f:
                        data = json.load(f)

                    # Find the patient record
                    patient_record = next((item for item in data if item['patient_id'] == selected_id), None)

                    if patient_record:
                        # Create tabs for better organization of data
                        tabs = st.tabs([
                            "Full Patient Record",
                            "Clinical Summary",
                            "Diagnostic Information",
                            "Treatment Plan",
                            "Conversation"
                        ])
                        with tabs[0]:
                            # Display full JSON data
                            st.json(patient_record.get('structured_data', {}))

                        with tabs[1]:
                            # Display clinical summary
                            clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})

                            if clinical_summary:
                                # Active symptoms
                                st.subheader("Active Symptoms")
                                symptoms = clinical_summary.get('ActiveSymptoms', [])
                                if symptoms:
                                    for symptom in symptoms:
                                        st.write(f"- {symptom}")
                                else:
                                    st.write("No active symptoms recorded")

                                # Negative findings
                                st.subheader("Negative Findings")
                                neg_findings = clinical_summary.get('NegativeFindings', [])
                                if neg_findings:
                                    for finding in neg_findings:
                                        st.write(f"- {finding}")
                                else:
                                    st.write("No negative findings recorded")

                                # Narrative summary if available
                                narrative = patient_record.get('structured_data', {}).get('SummaryNarrative', {})
                                if narrative:
                                    st.subheader("Clinical Narrative")

                                    # Clinical course
                                    course = narrative.get('ClinicalCourseProgression', '')
                                    if course:
                                        st.write(f"**Clinical Course:** {course}")

                                    # Diagnostic journey
                                    journey = narrative.get('DiagnosticJourney', '')
                                    if journey:
                                        st.write(f"**Diagnostic Journey:** {journey}")

                                    # Treatment response
                                    response = narrative.get('TreatmentResponse', '')
                                    if response:
                                        st.write(f"**Treatment Response:** {response}")

                                    # Ongoing concerns
                                    concerns = narrative.get('OngoingConcerns', '')
                                    if concerns:
                                        st.write(f"**Ongoing Concerns:** {concerns}")
                            else:
                                st.write("No clinical summary available")

                        with tabs[2]:
                            # Display diagnostic information
                            st.subheader("Diagnostic Conclusions")
                            diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions',
                                                                                             [])
                            if diag_conclusions:
                                for conclusion in diag_conclusions:
                                    st.write(f"- {conclusion}")
                            else:
                                st.write("No diagnostic conclusions available")

                            # Diagnostic evidence
                            st.subheader("Diagnostic Evidence")
                            diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})

                            # Imaging findings
                            img_findings = diag_evidence.get('ImagingFindings', [])
                            if img_findings:
                                st.write("**Imaging Findings:**")
                                for finding in img_findings:
                                    st.write(f"- {finding}")

                            # Lab results
                            lab_results = diag_evidence.get('LabResults', [])
                            if lab_results:
                                st.write("**Laboratory Results:**")
                                for result in lab_results:
                                    st.write(f"- {result}")

                            # Pathology findings
                            path_findings = diag_evidence.get('PathologyFindings', [])
                            if path_findings:
                                st.write("**Pathology Findings:**")
                                for finding in path_findings:
                                    st.write(f"- {finding}")

                        with tabs[3]:
                            # Display treatment plan information

                            # Therapeutic interventions
                            st.subheader("Therapeutic Interventions")
                            therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})

                            # Medications
                            medications = therapies.get('Medications', [])
                            if medications:
                                st.write("**Medications:**")
                                for med in medications:
                                    st.write(f"- {med}")
                            else:
                                st.write("No medications recorded")

                            # Procedures
                            procedures = therapies.get('Procedures', [])
                            if procedures:
                                st.write("**Procedures:**")
                                for proc in procedures:
                                    st.write(f"- {proc}")
                            else:
                                st.write("No procedures recorded")

                            # Follow-up plan
                            st.subheader("Follow-up Plan")
                            followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})

                            if followup:
                                # Planned consultations
                                consultations = followup.get('PlannedConsultations', [])
                                if consultations:
                                    st.write("**Planned Consultations:**")
                                    for consult in consultations:
                                        st.write(f"- {consult}")

                                # Scheduled tests
                                tests = followup.get('ScheduledTests', [])
                                if tests:
                                    st.write("**Scheduled Tests:**")
                                    for test in tests:
                                        st.write(f"- {test}")

                                # Next appointment
                                appointments = followup.get('NextAppointmentDetails', [])
                                if appointments:
                                    st.write("**Next Appointment Details:**")
                                    for appt in appointments:
                                        st.write(f"- {appt}")
                            else:
                                st.write("No follow-up plan recorded")

                            # Chronic conditions
                            st.subheader("Chronic Conditions")
                            chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})

                            if chronic:
                                # Chronic diseases
                                diseases = chronic.get('ChronicDiseases', [])
                                if diseases:
                                    st.write("**Chronic Diseases:**")
                                    for disease in diseases:
                                        st.write(f"- {disease}")

                                # Comorbidities
                                comorbidities = chronic.get('Comorbidities', [])
                                if comorbidities:
                                    st.write("**Comorbidities:**")
                                    for comorbidity in comorbidities:
                                        st.write(f"- {comorbidity}")
                            else:
                                st.write("No chronic conditions recorded")

                            # Visit timeline
                            st.subheader("Visit Timeline")
                            timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
                            if timeline:
                                for visit in timeline:
                                    st.write(f"- {visit}")
                            else:
                                st.write("No visit timeline recorded")

                        with tabs[4]:
                            # Display original conversation
                            st.text_area("Full Clinical Text", patient_record.get('raw_text', 'No text available'),
                                         height=400)
                    else:
                        st.warning(f"Could not find detailed record for patient {selected_id}")

                except Exception as e:
                    st.error(f"Error loading patient data: {e}")
    else:
        st.error("No patient data available. Please add patient records first.")


def extract_gender(structured_data):
    """Extract gender from structured data"""
    if isinstance(structured_data, dict):
        demographics = structured_data.get('PatientDemographics', {})
        return demographics.get('Gender', 'Unknown')
    return 'Unknown'


def show_analytics_page():
    st.header("HEPABOT Analytics")

    # Load metadata
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Process data for analytics
    genders = []
    ages = []
    diagnoses = []

    for patient in data:
        if 'structured_data' in patient:
            demographics = patient['structured_data'].get('PatientDemographics', {})
            genders.append(demographics.get('Gender', 'Unknown'))

            # Handle age - ensure it's numeric
            age = demographics.get('Age', '')
            try:
                age = int(age)
                ages.append(age)
            except (ValueError, TypeError):
                pass

            diagnoses.append(demographics.get('Diagnosis', 'Unknown'))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Demographics")

        # Gender distribution
        gender_counts = pd.Series(genders).value_counts()
        st.bar_chart(gender_counts)

        # Age distribution
        if ages:
            st.subheader("Age Distribution")
            age_df = pd.DataFrame({'Age': ages})

            # Define your own bins
            min_age = int(min(ages)) // 10 * 10  # Floor to nearest 10
            max_age = int(max(ages)) // 10 * 10 + 10  # Ceil to nearest 10
            bins = list(range(min_age, max_age + 1, 10))  # [0,10,20,...]

            # Now cut using these bins
            age_bins = pd.cut(age_df['Age'], bins=bins, right=False)

            hist_values = pd.DataFrame(age_bins.value_counts().sort_index())
            hist_values.index = hist_values.index.map(
                lambda x: f"{int(x.left)}–{int(x.right - 1)}")  # Format as "0–9", "10–19", etc.

            st.bar_chart(hist_values)

    import altair as alt

    with col2:
        st.subheader("Diagnosis Distribution")

        # Count diagnoses
        diagnosis_counts = pd.Series(diagnoses).value_counts().head(10).reset_index()
        diagnosis_counts.columns = ['Diagnosis', 'Count']

        chart = alt.Chart(diagnosis_counts).mark_bar().encode(
            x=alt.X(
                'Diagnosis:N',
                sort='-y',
                axis=alt.Axis(
                    labelAngle=-45,
                    labelFontSize=12,
                    labelOverlap=False  # <-- Force ALL labels to show
                )
            ),
            y=alt.Y('Count:Q')
        ).properties(
            width=700,
            height=400
        )

        st.altair_chart(chart, use_container_width=True)


def show_generate_report_page():
    # Constants
    OUTPUT_JSON = input_json_path
    output_directory = "patient_records"  # Adjust as per your setup

    # Check and load vector database if it exists but isn't loaded
    if os.path.exists(DB_PATH) and not st.session_state.get('vector_db_created', False):
        st.session_state.vector_db_created = True
        try:
            vector_db = load_vector_db(
                persist_directory=DB_PATH,
                collection_name=COLLECTION_NAME
            )
            if vector_db:
                st.session_state.rag_chain = create_rag_chain(vector_db, MODEL_NAME)
        except Exception as e:
            st.error(f"Error loading vector database: {e}")

    st.header("🩺 Generate Patient Report")
    st.write("Upload a clinical conversation PDF or audio file to generate a structured medical report")

    # Tabs for upload options
    tab1, tab2 = st.tabs(["Upload Audio", "Upload PDF"])

    with tab2:
        uploaded_pdf = st.file_uploader("Upload a clinical conversation PDF", type=["pdf"], key="pdf_uploader")

    with tab1:
        uploaded_audio = st.file_uploader(
            "Upload an audio file",
            type=["mp3", "wav", "m4a", "flac", "aac", "ogg"],
            key="audio_uploader"
        )

        if uploaded_audio:
            st.audio(uploaded_audio, format=f"audio/{uploaded_audio.name.split('.')[-1]}")

    if uploaded_pdf or uploaded_audio:
        # Determine which file to process
        if uploaded_pdf:
            # Save PDF to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as patient_file:
                patient_file.write(uploaded_pdf.read())
                tmp_path = patient_file.name
        elif uploaded_audio:
            # Process audio to PDF
            with st.spinner("🔊 Transcribing audio..."):
                pdf_path, error = process_audio_to_pdf(uploaded_audio)
                if error:
                    st.error(f"❌ Error transcribing audio: {error}")
                    return
                tmp_path = pdf_path
                # st.success("✅ Audio transcribed and converted to PDF")

        with st.spinner("🔍 Analyzing data..."):
            try:
                # Step 1: Process the PDF
                result = process_pdf(tmp_path)

                if not result:
                    st.error("❌ Failed to process PDF. No data extracted.")
                    return

                # Step 2: Set patient_id to MRN - Check both structured data and raw text
                demographics = result.get('structured_data', {}).get('PatientDemographics', {})
                mrn = demographics.get('MRN', None)

                # If MRN not found in structured data, try to extract it from raw text
                if not mrn and 'raw_text' in result:
                    raw_text = result.get('raw_text', '')
                    mrn_match = re.search(r'[Mm][Rr][Nn]:?\s*(\d+)', raw_text)
                    if mrn_match:
                        mrn = mrn_match.group(1)
                        demographics['MRN'] = mrn
                        result['structured_data']['PatientDemographics'] = demographics
                        st.info(f"MRN extracted from raw text: {mrn}")

                if mrn:
                    result['patient_id'] = mrn
                    # st.success(f"Using MRN as patient ID: {mrn}")
                else:
                    st.warning("No MRN found in patient data. Using default patient ID.")
                    result['patient_id'] = result.get('patient_id', f"patient_{int(time.time())}")

                # Step 3: Append to JSON file
                if os.path.exists(OUTPUT_JSON):
                    with open(OUTPUT_JSON, "r") as f:
                        existing_data = json.load(f)
                else:
                    existing_data = []

                # Check if this patient already exists
                existing_ids = [item.get('patient_id') for item in existing_data]
                if result['patient_id'] in existing_ids:
                    for i, item in enumerate(existing_data):
                        if item['patient_id'] == result['patient_id']:
                            existing_data[i] = result
                            st.info(f"⚠️ Updated existing record for patient {result['patient_id']}")
                else:
                    existing_data.append(result)
                    # st.success(f"✅ Added new patient record: {result['patient_id']}")

                # Save to combined JSON file
                with open(OUTPUT_JSON, "w") as f:
                    json.dump(existing_data, f, indent=2)

                # Step 4: Create individual JSON files
                os.makedirs(output_directory, exist_ok=True)
                save_patient_records(existing_data, output_directory)

                # Step 5: Add to vector database
                try:
                    collection = get_vector_db()
                    add_patient_record(collection, result)
                    # st.success("✅ Patient record added to vector database")
                except Exception as e:
                    st.error(f"❌ Error adding to vector database: {e}")

                # Step 6: Generate and display results
                st.subheader("📋 Generated Patient Report")

                # Generate PDF report and provide download button
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as report_file:
                    generate_pdf_report(result, report_file.name)
                    with open(report_file.name, "rb") as f:
                        st.download_button(
                            label="📥 Download Patient Report PDF",
                            data=f,
                            file_name=f"patient_report_{result['patient_id']}.pdf",
                            mime="application/pdf",
                            key=f"download_pdf_{result['patient_id']}"
                        )

                # Step 7: Display extracted data
                st.subheader("Extracted Patient Data")
                tabs = st.tabs(["Full Report", "Summary"])

                with tabs[1]:
                    demographics = result.get('structured_data', {}).get('PatientDemographics', {})
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("MRN", demographics.get('MRN', 'N/A'))
                    with col2:
                        st.metric("Age", demographics.get('Age', 'N/A'))
                    with col3:
                        st.metric("Gender", demographics.get('Gender', 'N/A'))
                    st.write(f"**Diagnosis:** {demographics.get('Diagnosis', 'N/A')}")
                    st.write(
                        f"**Summary:** {result.get('structured_data', {}).get('SummaryNarrative', {}).get('ClinicalCourseProgression', 'N/A')}")

                with tabs[0]:
                    st.json(result.get('structured_data', {}))

                # Step 8: Generate diagnosis using RAG pipeline
                if st.session_state.get('rag_chain', None):
                    with st.spinner("🩺 Generating AI-assisted diagnosis..."):
                        try:
                            demographics = result.get('structured_data', {}).get('PatientDemographics', {})
                            age = demographics.get('Age', 'unknown')
                            gender = demographics.get('Gender', 'unknown')

                            # Fix the error: Accessing dictionary values properly
                            summary_report = result.get('structured_data', {}).get('SummaryNarrative', {}).get(
                                'ClinicalCourseProgression', 'no symptoms provided')

                            # Properly access DiagnosticEvidence - get() with no arguments was causing the error
                            diagnostic_evidence = result.get('structured_data', {}).get('DiagnosticEvidence', {})
                            # Convert to string or use specific fields if available
                            diagnostic_evidence_str = str(
                                diagnostic_evidence) if diagnostic_evidence else "No diagnostic evidence available"

                            # Similarly fix ClinicalSummary access
                            clinical_summary = result.get('structured_data', {}).get('ClinicalSummary', {})
                            clinical_summary_str = str(
                                clinical_summary) if clinical_summary else "No clinical summary available"

                            question = (
                                f"Patient is {age} years old, {gender}, showing symptoms: {clinical_summary_str}, his Lab test include"
                                f" {diagnostic_evidence_str}, and his overall summary report is {summary_report} What is the likely diagnosis?")

                            # Get diagnosis from RAG chain
                            diagnosis = ask_question(st.session_state.rag_chain, question)

                            # Display diagnosis
                            st.subheader("AI-Assisted Diagnosis")
                            st.markdown(diagnosis)

                            # Store diagnosis in session state for potential reuse
                            # st.session_state.last_response = diagnosis

                            # Provide download button for diagnosis
                            col1, _ = st.columns(2)
                            with col1:
                                if st.button("Download Diagnosis"):
                                    download_data = diagnosis.encode()
                                    st.markdown(
                                        get_download_link(
                                            download_data,
                                            f"diagnosis_{result['patient_id']}.txt",
                                            "Download Diagnosis Result"
                                        ),
                                        unsafe_allow_html=True
                                    )
                        except Exception as e:
                            st.error(f"❌ Error generating diagnosis: {str(e)}")
                else:
                    st.warning("No vector database or RAG chain available to generate diagnosis.")

                # Step 9: Refresh metadata cache
                st.session_state.refresh_data = True
                st.cache_data.clear()

                # Add navigation button to patient browser
                if st.button("View in Patient Browser"):
                    st.session_state.page = "browser"
                    st.experimental_rerun()

            except Exception as e:
                st.error(f"❌ Error processing PDF: {e}")
            finally:
                # Clean up temporary file
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

def generate_pdf_report(patient_data, output_path):
    """Generateate a PDF report for the patient data."""
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y_position = height - 50
    line_height = 14

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, f"Patient Report - ID: {patient_data.get('patient_id', 'Unknown')}")
    y_position -= 30

    # Demographics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Patient Demographics")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    demographics = patient_data.get('structured_data', {}).get('PatientDemographics', {})
    for key, value in demographics.items():
        c.drawString(60, y_position, f"{key}: {value}")
        y_position -= line_height
    y_position -= 10

    # Clinical Summary
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Clinical Summary")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    clinical_summary = patient_data.get('structured_data', {}).get('ClinicalSummary', {})
    symptoms = clinical_summary.get('ActiveSymptoms', [])
    if symptoms:
        c.drawString(60, y_position, "Active Symptoms:")
        y_position -= line_height
        for symptom in symptoms:
            c.drawString(70, y_position, f"- {symptom}")
            y_position -= line_height
    neg_findings = clinical_summary.get('NegativeFindings', [])
    if neg_findings:
        c.drawString(60, y_position, "Negative Findings:")
        y_position -= line_height
        for finding in neg_findings:
            c.drawString(70, y_position, f"- {finding}")
            y_position -= line_height
    y_position -= 10

    # Diagnostic Information
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Diagnostic Information")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    diag_conclusions = patient_data.get('structured_data', {}).get('DiagnosticConclusions', [])
    if diag_conclusions:
        c.drawString(60, y_position, "Diagnostic Conclusions:")
        y_position -= line_height
        for conclusion in diag_conclusions:
            c.drawString(70, y_position, f"- {conclusion}")
            y_position -= line_height
    y_position -= 10

    # Treatment Plan
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Treatment Plan")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    therapies = patient_data.get('structured_data', {}).get('TherapeuticInterventions', {})
    medications = therapies.get('Medications', [])
    if medications:
        c.drawString(60, y_position, "Medications:")
        y_position -= line_height
        for med in medications:
            c.drawString(70, y_position, f"- {med}")
            y_position -= line_height
    procedures = therapies.get('Procedures', [])
    if procedures:
        c.drawString(60, y_position, "Procedures:")
        y_position -= line_height
        for proc in procedures:
            c.drawString(70, y_position, f"- {proc}")
            y_position -= line_height

    c.showPage()
    c.save()


if __name__ == "__main__":
    main()


# import streamlit as st
# import json
# import pandas as pd
# import os
# import tempfile
# from pathlib import Path
# from enhanced_extraction import extract_metadata, load_conversations_from_pdf, process_pdf
# import re
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfgen import canvas
# import time
#
# # Import the vector_db functions
# from vector_db import initialize_vector_db, add_patient_record, search_records, delete_patient_record
#
# input_json_path = "doctor_patient_data_80.json"
# output_directory = "split_patient_files"
# # Set page configuration
# st.set_page_config(
#     page_title="Medical Records System",
#     page_icon="🏥",
#     layout="wide"
# )
#
#
# # Load metadata of all patient records
# @st.cache_data
# def load_metadata(_refresh=False):
#     try:
#         with open(input_json_path, "r") as f:
#             data = json.load(f)
#
#         metadata_list = []
#         for entry in data:
#             if 'structured_data' in entry and 'patient_id' in entry:
#                 meta = entry['structured_data'].get('PatientDemographics', {})
#                 meta['patient_id'] = entry['patient_id']
#                 metadata_list.append(meta)
#             else:
#                 print(f"Skipping entry missing required fields: {entry}")
#
#         return metadata_list
#     except Exception as e:
#         print(f"Error loading metadata: {e}")
#         return []
#
#
# # Function to load and initialize the vector database
# @st.cache_resource
# def get_vector_db():
#     collection = initialize_vector_db()
#     return collection
#
#
# # Function to format search results for display
# def format_search_results(results):
#     formatted_results = []
#
#     if results and 'documents' in results and results['documents']:
#         for i, doc in enumerate(results['documents'][0]):
#             metadata = results['metadatas'][0][i] if 'metadatas' in results else {}
#
#             formatted_results.append({
#                 "patient_id": metadata.get("patient_id", "Unknown"),
#                 "gender": metadata.get("gender", "Unknown"),
#                 "age": metadata.get("age", "Unknown"),
#                 "mrn": metadata.get("mrn", "Unknown"),
#                 "diagnosis": metadata.get("diagnosis", "Unknown"),
#                 "content": doc
#             })
#
#     return formatted_results
#
#
# # Function to delete a patient record
# def delete_patient(patient_id):
#     try:
#         # Step 1: Remove from JSON file
#         if os.path.exists(input_json_path):
#             with open(input_json_path, "r") as f:
#                 data = json.load(f)
#             data = [item for item in data if item.get('patient_id') != patient_id]
#             with open(input_json_path, "w") as f:
#                 json.dump(data, f, indent=2)
#
#         # Step 2: Remove individual JSON file
#         individual_file = os.path.join(output_directory, f"{patient_id}.json")
#         if os.path.exists(individual_file):
#             os.remove(individual_file)
#
#         # Step 3: Remove from vector database
#         collection = get_vector_db()
#         delete_patient_record(collection, patient_id)
#
#         # Step 4: Refresh metadata cache
#         st.session_state.refresh_data = True
#         st.cache_data.clear()
#
#         st.success(f"✅ Patient record {patient_id} deleted successfully")
#     except Exception as e:
#         st.error(f"❌ Error deleting patient record {patient_id}: {e}")
#
#
# # Main function for the app
# def main():
#     st.title("🏥 Medical Records System")
#
#     # Initialize session state
#     if 'page' not in st.session_state:
#         st.session_state.page = "search"
#     if 'refresh_data' not in st.session_state:
#         st.session_state.refresh_data = False
#
#     # Sidebar for navigation
#     with st.sidebar:
#         st.title("Navigation")
#         if st.button("Diagnosis Search", use_container_width=True):
#             st.session_state.page = "search"
#         if st.button("Patient Browser", use_container_width=True):
#             st.session_state.page = "browser"
#         if st.button("Analytics", use_container_width=True):
#             st.session_state.page = "analytics"
#         if st.button("Generate Report", use_container_width=True):
#             st.session_state.page = "generate_report"
#
#     # Page selection
#     if st.session_state.page == "search":
#         show_search_page()
#     elif st.session_state.page == "browser":
#         show_browser_page()
#     elif st.session_state.page == "analytics":
#         show_analytics_page()
#     elif st.session_state.page == "generate_report":
#         show_generate_report_page()
#
#
# def show_search_page():
#     st.header("Diagnosis Search")
#     st.write("Search for patients with specific diagnoses or conditions")
#
#     # Clear cache if refresh is needed
#     if st.session_state.refresh_data:
#         st.cache_data.clear()
#         st.session_state.refresh_data = False
#
#     # Load metadata
#     metadata_list = load_metadata(_refresh=st.session_state.refresh_data)
#
#     # Get unique diagnoses from the metadata
#     all_diagnoses = []
#     for meta in metadata_list:
#         diagnosis = meta.get('Diagnosis', '')
#         if diagnosis and diagnosis not in all_diagnoses:
#             all_diagnoses.append(diagnosis)
#
#     # Sort diagnoses alphabetically
#     all_diagnoses.sort()
#
#     # Add an "All" option
#     all_diagnoses = ["All"] + all_diagnoses
#
#     # Create search interface
#     col1, col2 = st.columns([3, 1])
#
#     with col1:
#         search_query = st.text_input("Search by keyword in diagnosis:",
#                                      placeholder="E.g., cancer, diabetes, heart")
#
#     with col2:
#         selected_diagnosis = st.selectbox("Filter by diagnosis:", all_diagnoses)
#
#     # Filter options
#     with st.expander("Advanced Filters"):
#         col1, col2, col3 = st.columns(3)
#
#         with col1:
#             gender_filter = st.selectbox("Gender", ["Any", "Male", "Female"])
#
#         with col2:
#             age_range = st.slider("Age Range", 0, 100, (0, 100))
#
#         with col3:
#             sort_by = st.selectbox("Sort by", ["Diagnosis", "Age", "MRN"])
#
#     # Apply filters
#     filtered_patients = []
#
#     for patient in metadata_list:
#         diagnosis = patient.get('Diagnosis', '')
#         include = True
#
#         # Apply diagnosis filter
#         if selected_diagnosis != "All" and diagnosis != selected_diagnosis:
#             include = False
#
#         # Apply keyword search
#         if search_query and search_query.lower() not in diagnosis.lower():
#             include = False
#
#         # Apply gender filter
#         if gender_filter != "Any" and patient.get('Gender', '') != gender_filter:
#             include = False
#
#         # Apply age filter
#         try:
#             patient_age = int(patient.get('Age', 0))
#             if patient_age < age_range[0] or patient_age > age_range[1]:
#                 include = False
#         except (ValueError, TypeError):
#             pass
#
#         if include:
#             filtered_patients.append(patient)
#
#     # Sort the results
#     if sort_by == "Diagnosis":
#         filtered_patients.sort(key=lambda x: x.get('Diagnosis', ''))
#     elif sort_by == "Age":
#         # Convert age to int for sorting, with a default of 0 if conversion fails
#         filtered_patients.sort(key=lambda x: int(x.get('Age', 0)) if x.get('Age', '').isdigit() else 0)
#     elif sort_by == "MRN":
#         filtered_patients.sort(key=lambda x: x.get('MRN', ''))
#
#     # Display results
#     st.subheader(f"Search Results ({len(filtered_patients)} patients found)")
#
#     if filtered_patients:
#         # Create a DataFrame for display
#         df_display = []
#         for patient in filtered_patients:
#             df_display.append({
#                 "Patient ID": patient.get('patient_id', 'Unknown'),
#                 "MRN": patient.get('MRN', 'Unknown'),
#                 "Gender": patient.get('Gender', 'Unknown'),
#                 "Age": patient.get('Age', 'Unknown'),
#                 "Diagnosis": patient.get('Diagnosis', 'Unknown')
#             })
#
#         # Display as a table with ability to select a row
#         df = pd.DataFrame(df_display)
#         selected_indices = st.dataframe(df, use_container_width=True, height=300)
#
#         # Allow selecting a patient for detailed view
#         selected_patient_id = st.selectbox("Select a patient to view details:",
#                                            [''] + [p.get('patient_id', '') for p in filtered_patients])
#
#         if selected_patient_id:
#             # Find the selected patient in the JSON data
#             try:
#                 with open(input_json_path, "r") as f:
#                     all_data = json.load(f)
#
#                 # Find the selected patient record
#                 patient_record = next((item for item in all_data if item.get('patient_id') == selected_patient_id),
#                                       None)
#
#                 if patient_record:
#                     # Display patient details
#                     st.subheader("Patient Details")
#
#                     # Get demographics
#                     demographics = patient_record.get('structured_data', {}).get('PatientDemographics', {})
#
#                     # Display in columns
#                     col1, col2, col3, col4 = st.columns(4)
#                     with col1:
#                         st.metric("MRN", demographics.get('MRN', 'N/A'))
#                     with col2:
#                         st.metric("Age", demographics.get('Age', 'N/A'))
#                     with col3:
#                         st.metric("Gender", demographics.get('Gender', 'N/A'))
#                     with col4:
#                         st.metric("Diagnosis", demographics.get('Diagnosis', 'N/A'))
#
#                     # Display tabs for different sections of structured data
#                     tabs = st.tabs(["Summary Report", "Clinical Summary", "Diagnostic Information", "Treatment Plan",
#                                     "Conversation"])
#                     with tabs[0]:
#                         summary_report = patient_record.get('structured_data', {})
#                         st.subheader("Summary Report")
#                         st.write(summary_report)
#
#                     with tabs[1]:
#                         clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})
#
#                         st.subheader("Clinical Summary")
#
#                         # Active symptoms
#                         st.write("**Active Symptoms:**")
#                         symptoms = clinical_summary.get('ActiveSymptoms', [])
#                         if symptoms:
#                             for symptom in symptoms:
#                                 st.write(f"- {symptom}")
#                         else:
#                             st.write("No active symptoms recorded")
#
#                         # Negative findings
#                         st.write("**Negative Findings:**")
#                         neg_findings = clinical_summary.get('NegativeFindings', [])
#                         if neg_findings:
#                             for finding in neg_findings:
#                                 st.write(f"- {finding}")
#                         else:
#                             st.write("No negative findings recorded")
#
#                     with tabs[2]:
#                         # Display diagnostic information
#                         st.subheader("Diagnostic Information")
#
#                         # Diagnostic conclusions
#                         diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions', [])
#                         if diag_conclusions:
#                             st.write("**Diagnostic Conclusions:**")
#                             for conclusion in diag_conclusions:
#                                 st.write(f"- {conclusion}")
#
#                         # Diagnostic evidence
#                         diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})
#
#                         # Imaging findings
#                         img_findings = diag_evidence.get('ImagingFindings', [])
#                         if img_findings:
#                             st.write("**Imaging Findings:**")
#                             for finding in img_findings:
#                                 st.write(f"- {finding}")
#
#                         # Lab results
#                         lab_results = diag_evidence.get('LabResults', [])
#                         if lab_results:
#                             st.write("**Laboratory Results:**")
#                             for result in lab_results:
#                                 st.write(f"- {result}")
#
#                         # Pathology findings
#                         path_findings = diag_evidence.get('PathologyFindings', [])
#                         if path_findings:
#                             st.write("**Pathology Findings:**")
#                             for finding in path_findings:
#                                 st.write(f"- {finding}")
#
#                         # Chronic conditions
#                         chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})
#
#                         # Chronic diseases
#                         diseases = chronic.get('ChronicDiseases', [])
#                         if diseases:
#                             st.write("**Chronic Diseases:**")
#                             for disease in diseases:
#                                 st.write(f"- {disease}")
#
#                         # Comorbidities
#                         comorbidities = chronic.get('Comorbidities', [])
#                         if comorbidities:
#                             st.write("**Comorbidities:**")
#                             for comorbidity in comorbidities:
#                                 st.write(f"- {comorbidity}")
#
#                     with tabs[3]:
#                         # Display treatment plan
#                         st.subheader("Treatment and Follow-up Plan")
#
#                         # Therapeutic interventions
#                         therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})
#
#                         # Medications
#                         medications = therapies.get('Medications', [])
#                         if medications:
#                             st.write("**Medications:**")
#                             for med in medications:
#                                 st.write(f"- {med}")
#
#                         # Procedures
#                         procedures = therapies.get('Procedures', [])
#                         if procedures:
#                             st.write("**Procedures:**")
#                             for proc in procedures:
#                                 st.write(f"- {proc}")
#
#                         # Follow-up plan
#                         followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})
#
#                         # Planned consultations
#                         consultations = followup.get('PlannedConsultations', [])
#                         if consultations:
#                             st.write("**Planned Consultations:**")
#                             for consult in consultations:
#                                 st.write(f"- {consult}")
#
#                         # Scheduled tests
#                         tests = followup.get('ScheduledTests', [])
#                         if tests:
#                             st.write("**Scheduled Tests:**")
#                             for test in tests:
#                                 st.write(f"- {test}")
#
#                         # Next appointment
#                         appointments = followup.get('NextAppointmentDetails', [])
#                         if appointments:
#                             st.write("**Next Appointment Details:**")
#                             for appt in appointments:
#                                 st.write(f"- {appt}")
#
#                         # Visit timeline
#                         timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
#                         if timeline:
#                             st.write("**Visit Timeline:**")
#                             for visit in timeline:
#                                 st.write(f"- {visit}")
#
#                     with tabs[4]:
#                         # Display raw conversation
#                         st.text_area("Raw Conversation Text",
#                                      patient_record.get('raw_text', 'No conversation text available'), height=400)
#
#                 else:
#                     st.warning("Patient record not found in the database")
#
#             except Exception as e:
#                 st.error(f"Error retrieving patient data: {e}")
#     else:
#         st.warning("No patients match your search criteria. Please adjust your filters.")
#
#
# def show_browser_page():
#     st.header("Patient Record Browser")
#
#     # Clear cache if refresh is needed
#     if st.session_state.refresh_data:
#         st.cache_data.clear()
#         st.session_state.refresh_data = False
#
#     # Load metadata with refresh
#     metadata_list = load_metadata(_refresh=True)
#
#     # Create dataframe
#     if metadata_list:
#         df = pd.DataFrame(metadata_list)
#
#         # Sort options and filter options
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             sort_option = st.selectbox(
#                 "Sort by:",
#                 ["patient_id", "Age", "Gender", "MRN", "Diagnosis"]
#             )
#
#         with col2:
#             filter_gender = st.selectbox(
#                 "Filter by gender:",
#                 ["All", "Male", "Female"]
#             )
#
#         with col3:
#             search_term = st.text_input("Search by MRN or diagnosis:")
#
#         # Apply filters
#         filtered_df = df.copy()
#
#         if filter_gender != "All":
#             filtered_df = filtered_df[filtered_df["Gender"] == filter_gender]
#
#         if search_term:
#             # Search in MRN and Diagnosis columns
#             search_mask = (
#                     filtered_df["MRN"].astype(str).str.contains(search_term, case=False, na=False) |
#                     filtered_df["Diagnosis"].astype(str).str.contains(search_term, case=False, na=False)
#             )
#             filtered_df = filtered_df[search_mask]
#
#         # Sort dataframe
#         if sort_option in filtered_df.columns:
#             filtered_df = filtered_df.sort_values(by=sort_option)
#
#         # Display table with delete option
#         st.subheader("Patient Records")
#         # Create a new DataFrame for display with a delete button column
#         display_df = filtered_df[["patient_id", "MRN", "Gender", "Age", "Diagnosis"]].copy()
#         for idx, row in display_df.iterrows():
#             col1, col2 = st.columns([4, 1])
#             with col1:
#                 st.write(f"**{row['patient_id']}** | MRN: {row['MRN']} | Gender: {row['Gender']} | Age: {row['Age']} | Diagnosis: {row['Diagnosis']}")
#             with col2:
#                 if st.button("🗑️ Delete", key=f"delete_{row['patient_id']}"):
#                     delete_patient(row['patient_id'])
#                     st.experimental_rerun()  # Rerun to refresh the UI
#
#         # Patient detail view
#         st.subheader("Patient Detail View")
#
#         # Select patient by ID
#         selected_id = st.selectbox("Select Patient ID", sorted([m['patient_id'] for m in metadata_list]))
#
#         if selected_id:
#             # Find patient in metadata
#             patient_data = next((m for m in metadata_list if m['patient_id'] == selected_id), None)
#
#             if patient_data:
#                 # Display patient info
#                 col1, col2, col3 = st.columns(3)
#                 with col1:
#                     st.metric("MRN", patient_data.get('MRN', 'N/A'))
#                 with col2:
#                     st.metric("Age", patient_data.get('Age', 'N/A'))
#                 with col3:
#                     st.metric("Gender", patient_data.get('Gender', 'N/A'))
#
#                 st.subheader(f"Diagnosis: {patient_data.get('Diagnosis', 'N/A')}")
#
#                 # Load full patient data from combined JSON
#                 try:
#                     with open(input_json_path, "r") as f:
#                         data = json.load(f)
#
#                     # Find the patient record
#                     patient_record = next((item for item in data if item['patient_id'] == selected_id), None)
#
#                     if patient_record:
#                         # Create tabs for better organization of data
#                         tabs = st.tabs([
#                             "Full Patient Record",
#                             "Clinical Summary",
#                             "Diagnostic Information",
#                             "Treatment Plan",
#                             "Conversation"
#                         ])
#                         with tabs[0]:
#                             # Display full JSON data
#                             st.json(patient_record.get('structured_data', {}))
#
#                         with tabs[1]:
#                             # Display clinical summary
#                             clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})
#
#                             if clinical_summary:
#                                 # Active symptoms
#                                 st.subheader("Active Symptoms")
#                                 symptoms = clinical_summary.get('ActiveSymptoms', [])
#                                 if symptoms:
#                                     for symptom in symptoms:
#                                         st.write(f"- {symptom}")
#                                 else:
#                                     st.write("No active symptoms recorded")
#
#                                 # Negative findings
#                                 st.subheader("Negative Findings")
#                                 neg_findings = clinical_summary.get('NegativeFindings', [])
#                                 if neg_findings:
#                                     for finding in neg_findings:
#                                         st.write(f"- {finding}")
#                                 else:
#                                     st.write("No negative findings recorded")
#
#                                 # Narrative summary if available
#                                 narrative = patient_record.get('structured_data', {}).get('SummaryNarrative', {})
#                                 if narrative:
#                                     st.subheader("Clinical Narrative")
#
#                                     # Clinical course
#                                     course = narrative.get('ClinicalCourseProgression', '')
#                                     if course:
#                                         st.write(f"**Clinical Course:** {course}")
#
#                                     # Diagnostic journey
#                                     journey = narrative.get('DiagnosticJourney', '')
#                                     if journey:
#                                         st.write(f"**Diagnostic Journey:** {journey}")
#
#                                     # Treatment response
#                                     response = narrative.get('TreatmentResponse', '')
#                                     if response:
#                                         st.write(f"**Treatment Response:** {response}")
#
#                                     # Ongoing concerns
#                                     concerns = narrative.get('OngoingConcerns', '')
#                                     if concerns:
#                                         st.write(f"**Ongoing Concerns:** {concerns}")
#                             else:
#                                 st.write("No clinical summary available")
#
#                         with tabs[2]:
#                             # Display diagnostic information
#                             st.subheader("Diagnostic Conclusions")
#                             diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions',
#                                                                                              [])
#                             if diag_conclusions:
#                                 for conclusion in diag_conclusions:
#                                     st.write(f"- {conclusion}")
#                             else:
#                                 st.write("No diagnostic conclusions available")
#
#                             # Diagnostic evidence
#                             st.subheader("Diagnostic Evidence")
#                             diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})
#
#                             # Imaging findings
#                             img_findings = diag_evidence.get('ImagingFindings', [])
#                             if img_findings:
#                                 st.write("**Imaging Findings:**")
#                                 for finding in img_findings:
#                                     st.write(f"- {finding}")
#
#                             # Lab results
#                             lab_results = diag_evidence.get('LabResults', [])
#                             if lab_results:
#                                 st.write("**Laboratory Results:**")
#                                 for result in lab_results:
#                                     st.write(f"- {result}")
#
#                             # Pathology findings
#                             path_findings = diag_evidence.get('PathologyFindings', [])
#                             if path_findings:
#                                 st.write("**Pathology Findings:**")
#                                 for finding in path_findings:
#                                     st.write(f"- {finding}")
#
#                         with tabs[3]:
#                             # Display treatment plan information
#
#                             # Therapeutic interventions
#                             st.subheader("Therapeutic Interventions")
#                             therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})
#
#                             # Medications
#                             medications = therapies.get('Medications', [])
#                             if medications:
#                                 st.write("**Medications:**")
#                                 for med in medications:
#                                     st.write(f"- {med}")
#                             else:
#                                 st.write("No medications recorded")
#
#                             # Procedures
#                             procedures = therapies.get('Procedures', [])
#                             if procedures:
#                                 st.write("**Procedures:**")
#                                 for proc in procedures:
#                                     st.write(f"- {proc}")
#                             else:
#                                 st.write("No procedures recorded")
#
#                             # Follow-up plan
#                             st.subheader("Follow-up Plan")
#                             followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})
#
#                             if followup:
#                                 # Planned consultations
#                                 consultations = followup.get('PlannedConsultations', [])
#                                 if consultations:
#                                     st.write("**Planned Consultations:**")
#                                     for consult in consultations:
#                                         st.write(f"- {consult}")
#
#                                 # Scheduled tests
#                                 tests = followup.get('ScheduledTests', [])
#                                 if tests:
#                                     st.write("**Scheduled Tests:**")
#                                     for test in tests:
#                                         st.write(f"- {test}")
#
#                                 # Next appointment
#                                 appointments = followup.get('NextAppointmentDetails', [])
#                                 if appointments:
#                                     st.write("**Next Appointment Details:**")
#                                     for appt in appointments:
#                                         st.write(f"- {appt}")
#                             else:
#                                 st.write("No follow-up plan recorded")
#
#                             # Chronic conditions
#                             st.subheader("Chronic Conditions")
#                             chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})
#
#                             if chronic:
#                                 # Chronic diseases
#                                 diseases = chronic.get('ChronicDiseases', [])
#                                 if diseases:
#                                     st.write("**Chronic Diseases:**")
#                                     for disease in diseases:
#                                         st.write(f"- {disease}")
#
#                                 # Comorbidities
#                                 comorbidities = chronic.get('Comorbidities', [])
#                                 if comorbidities:
#                                     st.write("**Comorbidities:**")
#                                     for comorbidity in comorbidities:
#                                         st.write(f"- {comorbidity}")
#                             else:
#                                 st.write("No chronic conditions recorded")
#
#                             # Visit timeline
#                             st.subheader("Visit Timeline")
#                             timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
#                             if timeline:
#                                 for visit in timeline:
#                                     st.write(f"- {visit}")
#                             else:
#                                 st.write("No visit timeline recorded")
#
#                         with tabs[4]:
#                             # Display original conversation
#                             st.text_area("Full Clinical Text", patient_record.get('raw_text', 'No text available'),
#                                          height=400)
#                     else:
#                         st.warning(f"Could not find detailed record for patient {selected_id}")
#
#                 except Exception as e:
#                     st.error(f"Error loading patient data: {e}")
#     else:
#         st.error("No patient data available. Please add patient records first.")
#
#
# def extract_gender(structured_data):
#     """Extract gender from structured data"""
#     if isinstance(structured_data, dict):
#         demographics = structured_data.get('PatientDemographics', {})
#         return demographics.get('Gender', 'Unknown')
#     return 'Unknown'
#
#
# def show_analytics_page():
#     st.header("Medical Records Analytics")
#
#     # Load metadata
#     with open(input_json_path, "r") as f:
#         data = json.load(f)
#
#     # Process data for analytics
#     genders = []
#     ages = []
#     diagnoses = []
#
#     for patient in data:
#         if 'structured_data' in patient:
#             demographics = patient['structured_data'].get('PatientDemographics', {})
#             genders.append(demographics.get('Gender', 'Unknown'))
#
#             # Handle age - ensure it's numeric
#             age = demographics.get('Age', '')
#             try:
#                 age = int(age)
#                 ages.append(age)
#             except (ValueError, TypeError):
#                 pass
#
#             diagnoses.append(demographics.get('Diagnosis', 'Unknown'))
#
#     col1, col2 = st.columns(2)
#
#     with col1:
#         st.subheader("Patient Demographics")
#
#         # Gender distribution
#         gender_counts = pd.Series(genders).value_counts()
#         st.bar_chart(gender_counts)
#
#         # Age distribution
#         if ages:
#             st.subheader("Age Distribution")
#             age_df = pd.DataFrame({'Age': ages})
#
#             # Define your own bins
#             min_age = int(min(ages)) // 10 * 10  # Floor to nearest 10
#             max_age = int(max(ages)) // 10 * 10 + 10  # Ceil to nearest 10
#             bins = list(range(min_age, max_age + 1, 10))  # [0,10,20,...]
#
#             # Now cut using these bins
#             age_bins = pd.cut(age_df['Age'], bins=bins, right=False)
#
#             hist_values = pd.DataFrame(age_bins.value_counts().sort_index())
#             hist_values.index = hist_values.index.map(
#                 lambda x: f"{int(x.left)}–{int(x.right - 1)}")  # Format as "0–9", "10–19", etc.
#
#             st.bar_chart(hist_values)
#
#     import altair as alt
#
#     with col2:
#         st.subheader("Diagnosis Distribution")
#
#         # Count diagnoses
#         diagnosis_counts = pd.Series(diagnoses).value_counts().head(10).reset_index()
#         diagnosis_counts.columns = ['Diagnosis', 'Count']
#
#         chart = alt.Chart(diagnosis_counts).mark_bar().encode(
#             x=alt.X(
#                 'Diagnosis:N',
#                 sort='-y',
#                 axis=alt.Axis(
#                     labelAngle=-45,
#                     labelFontSize=12,
#                     labelOverlap=False  # <-- Force ALL labels to show
#                 )
#             ),
#             y=alt.Y('Count:Q')
#         ).properties(
#             width=700,
#             height=400
#         )
#
#         st.altair_chart(chart, use_container_width=True)
#
#
# def generate_pdf_report(patient_data, output_path):
#     """Generate a PDF report for the patient data."""
#     c = canvas.Canvas(output_path, pagesize=letter)
#     width, height = letter
#     y_position = height - 50
#     line_height = 14
#
#     # Title
#     c.setFont("Helvetica-Bold", 16)
#     c.drawString(50, y_position, f"Patient Report - ID: {patient_data.get('patient_id', 'Unknown')}")
#     y_position -= 30
#
#     # Demographics
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, y_position, "Patient Demographics")
#     y_position -= line_height
#     c.setFont("Helvetica", 10)
#     demographics = patient_data.get('structured_data', {}).get('PatientDemographics', {})
#     for key, value in demographics.items():
#         c.drawString(60, y_position, f"{key}: {value}")
#         y_position -= line_height
#     y_position -= 10
#
#     # Clinical Summary
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, y_position, "Clinical Summary")
#     y_position -= line_height
#     c.setFont("Helvetica", 10)
#     clinical_summary = patient_data.get('structured_data', {}).get('ClinicalSummary', {})
#     symptoms = clinical_summary.get('ActiveSymptoms', [])
#     if symptoms:
#         c.drawString(60, y_position, "Active Symptoms:")
#         y_position -= line_height
#         for symptom in symptoms:
#             c.drawString(70, y_position, f"- {symptom}")
#             y_position -= line_height
#     neg_findings = clinical_summary.get('NegativeFindings', [])
#     if neg_findings:
#         c.drawString(60, y_position, "Negative Findings:")
#         y_position -= line_height
#         for finding in neg_findings:
#             c.drawString(70, y_position, f"- {finding}")
#             y_position -= line_height
#     y_position -= 10
#
#     # Diagnostic Information
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, y_position, "Diagnostic Information")
#     y_position -= line_height
#     c.setFont("Helvetica", 10)
#     diag_conclusions = patient_data.get('structured_data', {}).get('DiagnosticConclusions', [])
#     if diag_conclusions:
#         c.drawString(60, y_position, "Diagnostic Conclusions:")
#         y_position -= line_height
#         for conclusion in diag_conclusions:
#             c.drawString(70, y_position, f"- {conclusion}")
#             y_position -= line_height
#     y_position -= 10
#
#     # Treatment Plan
#     c.setFont("Helvetica-Bold", 12)
#     c.drawString(50, y_position, "Treatment Plan")
#     y_position -= line_height
#     c.setFont("Helvetica", 10)
#     therapies = patient_data.get('structured_data', {}).get('TherapeuticInterventions', {})
#     medications = therapies.get('Medications', [])
#     if medications:
#         c.drawString(60, y_position, "Medications:")
#         y_position -= line_height
#         for med in medications:
#             c.drawString(70, y_position, f"- {med}")
#             y_position -= line_height
#     procedures = therapies.get('Procedures', [])
#     if procedures:
#         c.drawString(60, y_position, "Procedures:")
#         y_position -= line_height
#         for proc in procedures:
#             c.drawString(70, y_position, f"- {proc}")
#             y_position -= line_height
#
#     c.showPage()
#     c.save()
#
#
# def show_generate_report_page():
#     # Constants
#     OUTPUT_JSON = input_json_path
#
#     st.header("🩺 Generate Patient Report")
#     st.write("Upload a clinical conversation PDF to generate a structured medical report")
#
#     uploaded_file = st.file_uploader("Upload a clinical conversation PDF", type=["pdf"])
#
#     if uploaded_file:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as patient_file:
#             patient_file.write(uploaded_file.read())
#             tmp_path = patient_file.name
#
#         with st.spinner("🔍 Analyzing data..."):
#             try:
#                 # Step 1: Process the uploaded PDF
#                 result = process_pdf(tmp_path)
#
#                 if not result:
#                     st.error("❌ Failed to process PDF. No data extracted.")
#                 else:
#                     # Step 2: Set patient_id to MRN
#                     demographics = result.get('structured_data', {}).get('PatientDemographics', {})
#                     mrn = demographics.get('MRN', None)
#                     if mrn:
#                         result['patient_id'] = mrn
#                     else:
#                         st.warning("No MRN found in patient data. Using default patient ID.")
#                         result['patient_id'] = result.get('patient_id', f"patient_{int(time.time())}")
#
#                     # Step 3: Append to JSON file
#                     if os.path.exists(OUTPUT_JSON):
#                         with open(OUTPUT_JSON, "r") as f:
#                             existing_data = json.load(f)
#                     else:
#                         existing_data = []
#
#                     # Check if this patient already exists
#                     existing_ids = [item.get('patient_id') for item in existing_data]
#                     if result['patient_id'] in existing_ids:
#                         # Update existing record
#                         for i, item in enumerate(existing_data):
#                             if item['patient_id'] == result['patient_id']:
#                                 existing_data[i] = result
#                                 st.info(f"⚠️ Updated existing record for patient {result['patient_id']}")
#                     else:
#                         # Add new record
#                         existing_data.append(result)
#                         st.success(f"✅ Added new patient record: {result['patient_id']}")
#
#                     # Save to combined JSON file
#                     with open(OUTPUT_JSON, "w") as f:
#                         json.dump(existing_data, f, indent=2)
#
#                     # Step 4: Create individual JSON files
#                     os.makedirs(output_directory, exist_ok=True)
#                     save_patient_records(existing_data, output_directory)
#
#                     # Step 5: Add to vector database
#                     try:
#                         collection = get_vector_db()
#                         add_patient_record(collection, result)
#                         st.success("✅ Patient record added to vector database")
#                     except Exception as e:
#                         st.error(f"❌ Error adding to vector database: {e}")
#
#                     # Step 6: Generate PDF report
#                     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as report_file:
#                         generate_pdf_report(result, report_file.name)
#                         with open(report_file.name, "rb") as f:
#                             st.download_button(
#                                 label="📥 Download Patient Report PDF",
#                                 data=f,
#                                 file_name=f"patient_report_{result['patient_id']}.pdf",
#                                 mime="application/pdf"
#                             )
#
#                     # Step 7: Display extracted data
#                     st.subheader("Extracted Patient Data")
#                     tabs = st.tabs(["Full Report", "Summary"])
#
#                     with tabs[1]:
#                         demographics = result.get('structured_data', {}).get('PatientDemographics', {})
#                         col1, col2, col3 = st.columns(3)
#                         with col1:
#                             st.metric("MRN", demographics.get('MRN', 'N/A'))
#                         with col2:
#                             st.metric("Age", demographics.get('Age', 'N/A'))
#                         with col3:
#                             st.metric("Gender", demographics.get('Gender', 'N/A'))
#                         st.write(f"**Diagnosis:** {demographics.get('Diagnosis', 'N/A')}")
#                         st.write(
#                             f"**Summary:** {result.get('structured_data', {}).get('SummaryNarrative', {}).get('ClinicalCourseProgression', 'N/A')}")
#
#                     with tabs[0]:
#                         st.json(result.get('structured_data', {}))
#
#                     # Step 8: Refresh metadata cache
#                     st.session_state.refresh_data = True
#                     st.cache_data.clear()
#
#             except Exception as e:
#                 st.error(f"❌ Error processing PDF: {e}")
#             finally:
#                 # Clean up temporary file
#                 try:
#                     os.unlink(tmp_path)
#                 except Exception:
#                     pass
#
#
# if __name__ == "__main__":
#     main()
