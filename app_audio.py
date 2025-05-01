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
import streamlit.components.v1 as components
import base64
import whisper
from pyannote.audio import Pipeline
import torch
import torchaudio

# Import the vector_db functions
from vector_db import initialize_vector_db, add_patient_record, search_records
from extract_each_patient_json import save_patient_records

input_json_path = "doctor_patient_data_80.json"
output_directory = "split_patient_files"

# Set page configuration
st.set_page_config(
    page_title="Medical Records System",
    page_icon="🏥",
    layout="wide"
)


# Transcription function using Whisper and pyannote.audio
@st.cache_resource
def load_transcription_models():
    # Load Whisper model (use 'base' for efficiency, or 'small'/'medium' for better accuracy)
    whisper_model = whisper.load_model("base")

    # Load pyannote.audio diarization pipeline
    # Replace 'your_hugging_face_token' with your actual Hugging Face token
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token="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"  # Or set HF_TOKEN environment variable
    )

    # Move to GPU if available
    if torch.cuda.is_available():
        whisper_model = whisper_model.to("cuda")
        diarization_pipeline = diarization_pipeline.to(torch.device("cuda"))

    return whisper_model, diarization_pipeline


def transcribe_audio(audio_path):
    """
    Transcribe audio file and segment into doctor/patient turns using Whisper and pyannote.audio.
    Returns a dictionary with raw_text and segmented_conversation.
    """
    try:
        # Load models
        whisper_model, diarization_pipeline = load_transcription_models()

        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to 16kHz if needed (required by Whisper and pyannote)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Save temporary WAV file for Whisper and pyannote
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            torchaudio.save(temp_audio.name, waveform, sample_rate)
            temp_audio_path = temp_audio.name

        # Transcribe audio using Whisper
        result = whisper_model.transcribe(temp_audio_path, language="en")
        raw_text = result["text"]

        # Perform diarization using pyannote.audio
        diarization = diarization_pipeline(temp_audio_path)

        # Align transcription segments with diarization
        segmented_conversation = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            # Find transcription segments that overlap with this speaker turn
            speaker_text = ""
            for segment in result["segments"]:
                if (segment["start"] >= turn.start and segment["end"] <= turn.end):
                    speaker_text += segment["text"] + " "

            # Assign speaker label (heuristic: assume SPEAKER_00 is Doctor, SPEAKER_01 is Patient)
            speaker_label = "Doctor" if speaker == "SPEAKER_00" else "Patient"

            if speaker_text.strip():
                segmented_conversation.append({
                    "speaker": speaker_label,
                    "text": speaker_text.strip()
                })

        # Format raw_text with speaker labels
        formatted_text = ""
        for segment in segmented_conversation:
            formatted_text += f"{segment['speaker']}: {segment['text']}\n"

        # Clean up temporary file
        os.unlink(temp_audio_path)

        return {
            "raw_text": formatted_text,
            "segmented_conversation": segmented_conversation
        }
    except Exception as e:
        raise Exception(f"Transcription failed: {e}")


# Load metadata of all patient records
@st.cache_data
def load_metadata(refresh=False):
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


# New function to process raw text input
def process_text(text, segmented_conversation=None):
    """
    Process raw text to generate structured patient data.
    Assumes extract_metadata can handle raw text input.
    """
    try:
        # Create a temporary text file with the raw text
        with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as txt_file:
            txt_file.write(text.encode('utf-8'))
            txt_path = txt_file.name

        # Use extract_metadata to process the text
        # Assuming extract_metadata can handle raw text and returns structured data
        structured_data = extract_metadata(txt_path, is_text=True)

        # Generate a unique patient ID (simple increment based on existing IDs)
        if os.path.exists(input_json_path):
            with open(input_json_path, "r") as f:
                existing_data = json.load(f)
                existing_ids = [int(item['patient_id'].split('_')[-1]) for item in existing_data if
                                item['patient_id'].startswith('patient_')]
                new_id = max(existing_ids) + 1 if existing_ids else 1
        else:
            new_id = 1

        patient_id = f"patient_{new_id:04d}"

        # Construct result
        result = {
            "patient_id": patient_id,
            "raw_text": text,
            "structured_data": structured_data,
            "segmented_conversation": segmented_conversation or []
        }

        return result
    except Exception as e:
        raise Exception(f"Error processing text: {e}")
    finally:
        try:
            os.unlink(txt_path)
        except Exception:
            pass


# Main function for the app
def main():
    st.title("🏥 Medical Records System")

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
        if st.button("Disease Diagnosis", use_container_width=True):
            st.session_state.page = "disease_diagnosis"

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
    elif st.session_state.page == "disease_diagnosis":
        show_disease_diagnosis_page()


def show_search_page():
    st.header("Diagnosis Search")
    st.write("Search for patients with specific diagnoses or conditions")

    # Load metadata
    metadata_list = load_metadata(refresh=st.session_state.refresh_data)

    # Reset refresh flag if it was set
    if st.session_state.refresh_data:
        st.session_state.refresh_data = False

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
        filtered_patients.sort(key=lambda x: int(x.get('Age', 0)) if x.get('Age', '').isdigit() else 0)
    elif sort_by == "MRN":
        filtered_patients.sort(key=lambda x: x.get('MRN', ''))

    # Display results
    st.subheader(f"Search Results ({len(filtered_patients)} patients found)")

    if filtered_patients:
        df_display = []
        for patient in filtered_patients:
            df_display.append({
                "Patient ID": patient.get('patient_id', 'Unknown'),
                "MRN": patient.get('MRN', 'Unknown'),
                "Gender": patient.get('Gender', 'Unknown'),
                "Age": patient.get('Age', 'Unknown'),
                "Diagnosis": patient.get('Diagnosis', 'Unknown')
            })

        df = pd.DataFrame(df_display)
        st.dataframe(df, use_container_width=True, height=300)

        selected_patient_id = st.selectbox("Select a patient to view details:",
                                           [''] + [p.get('patient_id', '') for p in filtered_patients])

        if selected_patient_id:
            try:
                with open(input_json_path, "r") as f:
                    all_data = json.load(f)

                patient_record = next((item for item in all_data if item.get('patient_id') == selected_patient_id),
                                      None)

                if patient_record:
                    st.subheader("Patient Details")
                    demographics = patient_record.get('structured_data', {}).get('PatientDemographics', {})
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("MRN", demographics.get('MRN', 'N/A'))
                    with col2:
                        st.metric("Age", demographics.get('Age', 'N/A'))
                    with col3:
                        st.metric("Gender", demographics.get('Gender', 'N/A'))
                    with col4:
                        st.metric("Diagnosis", demographics.get('Diagnosis', 'N/A'))

                    tabs = st.tabs(["Summary Report", "Clinical Summary", "Diagnostic Information", "Treatment Plan",
                                    "Conversation"])
                    with tabs[0]:
                        summary_report = patient_record.get('structured_data', {})
                        st.subheader("Summary Report")
                        st.write(summary_report)

                    with tabs[1]:
                        clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})
                        st.subheader("Clinical Summary")
                        st.write("**Active Symptoms:**")
                        symptoms = clinical_summary.get('ActiveSymptoms', [])
                        if symptoms:
                            for symptom in symptoms:
                                st.write(f"- {symptom}")
                        else:
                            st.write("No active symptoms recorded")
                        st.write("**Negative Findings:**")
                        neg_findings = clinical_summary.get('NegativeFindings', [])
                        if neg_findings:
                            for finding in neg_findings:
                                st.write(f"- {finding}")
                        else:
                            st.write("No negative findings recorded")

                    with tabs[2]:
                        st.subheader("Diagnostic Information")
                        diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions', [])
                        if diag_conclusions:
                            st.write("**Diagnostic Conclusions:**")
                            for conclusion in diag_conclusions:
                                st.write(f"- {conclusion}")
                        diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})
                        img_findings = diag_evidence.get('ImagingFindings', [])
                        if img_findings:
                            st.write("**Imaging Findings:**")
                            for finding in img_findings:
                                st.write(f"- {finding}")
                        lab_results = diag_evidence.get('LabResults', [])
                        if lab_results:
                            st.write("**Laboratory Results:**")
                            for result in lab_results:
                                st.write(f"- {result}")
                        path_findings = diag_evidence.get('PathologyFindings', [])
                        if path_findings:
                            st.write("**Pathology Findings:**")
                            for finding in path_findings:
                                st.write(f"- {finding}")
                        chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})
                        diseases = chronic.get('ChronicDiseases', [])
                        if diseases:
                            st.write("**Chronic Diseases:**")
                            for disease in diseases:
                                st.write(f"- {disease}")
                        comorbidities = chronic.get('Comorbidities', [])
                        if comorbidities:
                            st.write("**Comorbidities:**")
                            for comorbidity in comorbidities:
                                st.write(f"- {comorbidity}")

                    with tabs[3]:
                        st.subheader("Treatment and Follow-up Plan")
                        therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})
                        medications = therapies.get('Medications', [])
                        if medications:
                            st.write("**Medications:**")
                            for med in medications:
                                st.write(f"- {med}")
                        procedures = therapies.get('Procedures', [])
                        if procedures:
                            st.write("**Procedures:**")
                            for proc in procedures:
                                st.write(f"- {proc}")
                        followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})
                        consultations = followup.get('PlannedConsultations', [])
                        if consultations:
                            st.write("**Planned Consultations:**")
                            for consult in consultations:
                                st.write(f"- {consult}")
                        tests = followup.get('ScheduledTests', [])
                        if tests:
                            st.write("**Scheduled Tests:**")
                            for test in tests:
                                st.write(f"- {test}")
                        appointments = followup.get('NextAppointmentDetails', [])
                        if appointments:
                            st.write("**Next Appointment Details:**")
                            for appt in appointments:
                                st.write(f"- {appt}")
                        timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
                        if timeline:
                            st.write("**Visit Timeline:**")
                            for visit in timeline:
                                st.write(f"- {visit}")

                    with tabs[4]:
                        st.text_area("Raw Conversation Text",
                                     patient_record.get('raw_text', 'No conversation text available'), height=400)
                        if patient_record.get('segmented_conversation'):
                            st.subheader("Segmented Conversation")
                            for segment in patient_record.get('segmented_conversation', []):
                                st.write(f"**{segment['speaker']}:** {segment['text']}")

                else:
                    st.warning("Patient record not found in the database")
            except Exception as e:
                st.error(f"Error retrieving patient data: {e}")
    else:
        st.warning("No patients match your search criteria. Please adjust your filters.")


def show_browser_page():
    st.header("Patient Record Browser")
    metadata_list = load_metadata(refresh=st.session_state.refresh_data)
    if st.session_state.refresh_data:
        st.session_state.refresh_data = False

    if metadata_list:
        df = pd.DataFrame(metadata_list)
        col1, col2, col3 = st.columns(3)
        with col1:
            sort_option = st.selectbox("Sort by:", ["patient_id", "Age", "Gender", "MRN", "Diagnosis"])
        with col2:
            filter_gender = st.selectbox("Filter by gender:", ["All", "Male", "Female"])
        with col3:
            search_term = st.text_input("Search by MRN or diagnosis:")

        filtered_df = df.copy()
        if filter_gender != "All":
            filtered_df = filtered_df[filtered_df["Gender"] == filter_gender]
        if search_term:
            search_mask = (
                    filtered_df["MRN"].astype(str).str.contains(search_term, case=False, na=False) |
                    filtered_df["Diagnosis"].astype(str).str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        if sort_option in filtered_df.columns:
            filtered_df = filtered_df.sort_values(by=sort_option)

        st.dataframe(filtered_df, use_container_width=True)
        st.subheader("Patient Detail View")
        selected_id = st.selectbox("Select Patient ID", sorted([m['patient_id'] for m in metadata_list]))

        if selected_id:
            patient_data = next((m for m in metadata_list if m['patient_id'] == selected_id), None)
            if patient_data:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("MRN", patient_data.get('MRN', 'N/A'))
                with col2:
                    st.metric("Age", patient_data.get('Age', 'N/A'))
                with col3:
                    st.metric("Gender", patient_data.get('Gender', 'N/A'))
                st.subheader(f"Diagnosis: {patient_data.get('Diagnosis', 'N/A')}")
                try:
                    with open(input_json_path, "r") as f:
                        data = json.load(f)
                    patient_record = next((item for item in data if item['patient_id'] == selected_id), None)
                    if patient_record:
                        tabs = st.tabs(
                            ["Full Patient Record", "Clinical Summary", "Diagnostic Information", "Treatment Plan",
                             "Conversation"])
                        with tabs[0]:
                            st.json(patient_record.get('structured_data', {}))
                        with tabs[1]:
                            clinical_summary = patient_record.get('structured_data', {}).get('ClinicalSummary', {})
                            if clinical_summary:
                                st.subheader("Active Symptoms")
                                symptoms = clinical_summary.get('ActiveSymptoms', [])
                                if symptoms:
                                    for symptom in symptoms:
                                        st.write(f"- {symptom}")
                                else:
                                    st.write("No active symptoms recorded")
                                st.subheader("Negative Findings")
                                neg_findings = clinical_summary.get('NegativeFindings', [])
                                if neg_findings:
                                    for finding in neg_findings:
                                        st.write(f"- {finding}")
                                else:
                                    st.write("No negative findings recorded")
                                narrative = patient_record.get('structured_data', {}).get('SummaryNarrative', {})
                                if narrative:
                                    st.subheader("Clinical Narrative")
                                    course = narrative.get('ClinicalCourseProgression', '')
                                    if course:
                                        st.write(f"**Clinical Course:** {course}")
                                    journey = narrative.get('DiagnosticJourney', '')
                                    if journey:
                                        st.write(f"**Diagnostic Journey:** {journey}")
                                    response = narrative.get('TreatmentResponse', '')
                                    if response:
                                        st.write(f"**Treatment Response:** {response}")
                                    concerns = narrative.get('OngoingConcerns', '')
                                    if concerns:
                                        st.write(f"**Ongoing Concerns:** {concerns}")
                            else:
                                st.write("No clinical summary available")
                        with tabs[2]:
                            st.subheader("Diagnostic Conclusions")
                            diag_conclusions = patient_record.get('structured_data', {}).get('DiagnosticConclusions',
                                                                                             [])
                            if diag_conclusions:
                                for conclusion in diag_conclusions:
                                    st.write(f"- {conclusion}")
                            else:
                                st.write("No diagnostic conclusions available")
                            st.subheader("Diagnostic Evidence")
                            diag_evidence = patient_record.get('structured_data', {}).get('DiagnosticEvidence', {})
                            img_findings = diag_evidence.get('ImagingFindings', [])
                            if img_findings:
                                st.write("**Imaging Findings:**")
                                for finding in img_findings:
                                    st.write(f"- {finding}")
                            lab_results = diag_evidence.get('LabResults', [])
                            if lab_results:
                                st.write("**Laboratory Results:**")
                                for result in lab_results:
                                    st.write(f"- {result}")
                            path_findings = diag_evidence.get('PathologyFindings', [])
                            if path_findings:
                                st.write("**Pathology Findings:**")
                                for finding in path_findings:
                                    st.write(f"- {finding}")
                        with tabs[3]:
                            st.subheader("Therapeutic Interventions")
                            therapies = patient_record.get('structured_data', {}).get('TherapeuticInterventions', {})
                            medications = therapies.get('Medications', [])
                            if medications:
                                st.write("**Medications:**")
                                for med in medications:
                                    st.write(f"- {med}")
                            else:
                                st.write("No medications recorded")
                            procedures = therapies.get('Procedures', [])
                            if procedures:
                                st.write("**Procedures:**")
                                for proc in procedures:
                                    st.write(f"- {proc}")
                            else:
                                st.write("No procedures recorded")
                            st.subheader("Follow-up Plan")
                            followup = patient_record.get('structured_data', {}).get('Follow-upPlan', {})
                            if followup:
                                consultations = followup.get('PlannedConsultations', [])
                                if consultations:
                                    st.write("**Planned Consultations:**")
                                    for consult in consultations:
                                        st.write(f"- {consult}")
                                tests = followup.get('ScheduledTests', [])
                                if tests:
                                    st.write("**Scheduled Tests:**")
                                    for test in tests:
                                        st.write(f"- {test}")
                                appointments = followup.get('NextAppointmentDetails', [])
                                if appointments:
                                    st.write("**Next Appointment Details:**")
                                    for appt in appointments:
                                        st.write(f"- {appt}")
                            else:
                                st.write("No follow-up plan recorded")
                            st.subheader("Chronic Conditions")
                            chronic = patient_record.get('structured_data', {}).get('ChronicConditions', {})
                            if chronic:
                                diseases = chronic.get('ChronicDiseases', [])
                                if diseases:
                                    st.write("**Chronic Diseases:**")
                                    for disease in diseases:
                                        st.write(f"- {disease}")
                                comorbidities = chronic.get('Comorbidities', [])
                                if comorbidities:
                                    st.write("**Comorbidities:**")
                                    for comorbidity in comorbidities:
                                        st.write(f"- {comorbidity}")
                            else:
                                st.write("No chronic conditions recorded")
                            st.subheader("Visit Timeline")
                            timeline = patient_record.get('structured_data', {}).get('VisitTimeline', [])
                            if timeline:
                                for visit in timeline:
                                    st.write(f"- {visit}")
                            else:
                                st.write("No visit timeline recorded")
                        with tabs[4]:
                            st.text_area("Full Clinical Text", patient_record.get('raw_text', 'No text available'),
                                         height=400)
                            if patient_record.get('segmented_conversation'):
                                st.subheader("Segmented Conversation")
                                for segment in patient_record.get('segmented_conversation', []):
                                    st.write(f"**{segment['speaker']}:** {segment['text']}")
                    else:
                        st.warning(f"Could not find detailed record for patient {selected_id}")
                except Exception as e:
                    st.error(f"Error loading patient data: {e}")
    else:
        st.error("No patient data available. Please add patient records first.")


def extract_gender(structured_data):
    if isinstance(structured_data, dict):
        demographics = structured_data.get('PatientDemographics', {})
        return demographics.get('Gender', 'Unknown')
    return 'Unknown'


def show_analytics_page():
    st.header("Medical Records Analytics")
    with open(input_json_path, "r") as f:
        data = json.load(f)

    genders = []
    ages = []
    diagnoses = []

    for patient in data:
        if 'structured_data' in patient:
            demographics = patient['structured_data'].get('PatientDemographics', {})
            genders.append(demographics.get('Gender', 'Unknown'))
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
        gender_counts = pd.Series(genders).value_counts()
        st.bar_chart(gender_counts)
        if ages:
            st.subheader("Age Distribution")
            age_df = pd.DataFrame({'Age': ages})
            min_age = int(min(ages)) // 10 * 10
            max_age = int(max(ages)) // 10 * 10 + 10
            bins = list(range(min_age, max_age + 1, 10))
            age_bins = pd.cut(age_df['Age'], bins=bins, right=False)
            hist_values = pd.DataFrame(age_bins.value_counts().sort_index())
            hist_values.index = hist_values.index.map(lambda x: f"{int(x.left)}–{int(x.right - 1)}")
            st.bar_chart(hist_values)

    import altair as alt
    with col2:
        st.subheader("Diagnosis Distribution")
        diagnosis_counts = pd.Series(diagnoses).value_counts().head(10).reset_index()
        diagnosis_counts.columns = ['Diagnosis', 'Count']
        chart = alt.Chart(diagnosis_counts).mark_bar().encode(
            x=alt.X(
                'Diagnosis:N',
                sort='-y',
                axis=alt.Axis(labelAngle=-45, labelFontSize=12, labelOverlap=False)
            ),
            y=alt.Y('Count:Q')
        ).properties(width=700, height=400)
        st.altair_chart(chart, use_container_width=True)


def generate_pdf_report(patient_data, output_path):
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y_position = height - 50
    line_height = 14

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, f"Patient Report - ID: {patient_data.get('patient_id', 'Unknown')}")
    y_position -= 30

    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, y_position, "Patient Demographics")
    y_position -= line_height
    c.setFont("Helvetica", 10)
    demographics = patient_data.get('structured_data', {}).get('PatientDemographics', {})
    for key, value in demographics.items():
        c.drawString(60, y_position, f"{key}: {value}")
        y_position -= line_height
    y_position -= 10

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


def show_generate_report_page():
    OUTPUT_JSON = input_json_path
    st.header("🩺 Generate Patient Report")
    st.write(
        "Upload a clinical conversation PDF, audio file, or record a conversation to generate a structured medical report")

    input_type = st.radio("Select input type:", ["PDF", "Audio File", "Record Audio"])

    if input_type == "PDF":
        uploaded_file = st.file_uploader("Upload a clinical conversation PDF", type=["pdf"])
        if uploaded_file:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as patient_file:
                patient_file.write(uploaded_file.read())
                tmp_path = patient_file.name
            with st.spinner("🔍 Analyzing PDF..."):
                try:
                    result = process_pdf(tmp_path)
                    if not result:
                        st.error("❌ Failed to process PDF. No data extracted.")
                    else:
                        process_and_save_result(result)
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {e}")
                finally:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    elif input_type == "Audio File":
        uploaded_audio = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])
        if uploaded_audio:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                audio_file.write(uploaded_audio.read())
                tmp_audio_path = audio_file.name
            with st.spinner("🎙️ Transcribing audio..."):
                try:
                    transcription = transcribe_audio(tmp_audio_path)
                    if not transcription or not transcription.get("raw_text"):
                        st.error("❌ Failed to transcribe audio. No text extracted.")
                    else:
                        result = process_text(transcription["raw_text"], transcription["segmented_conversation"])
                        if result:
                            process_and_save_result(result)
                        else:
                            st.error("❌ Failed to process transcribed text.")
                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                finally:
                    try:
                        os.unlink(tmp_audio_path)
                    except Exception:
                        pass

    elif input_type == "Record Audio":
        st.write(
            "Use the microphone to record a clinical conversation. Click 'Start recording' to begin, and click again to stop.")

        # Check if streamlit's audio_recorder is available (Streamlit >= 1.18.0)
        try:
            audio_bytes = st.audio_recorder(
                pause_threshold=2.0,  # Automatically stop recording after 2 seconds of silence
                sample_rate=16000,  # 16kHz sample rate for speech recognition
                key="audio_recorder"
            )

            if audio_bytes:
                st.audio(audio_bytes, format="audio/wav")
                st.success("✅ Audio recorded successfully! Click the 'Process Recording' button below.")

                if st.button("Process Recording"):
                    with st.spinner("🎙️ Transcribing recorded audio..."):
                        try:
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as audio_file:
                                audio_file.write(audio_bytes)
                                tmp_audio_path = audio_file.name

                            transcription = transcribe_audio(tmp_audio_path)
                            if not transcription or not transcription.get("raw_text"):
                                st.error("❌ Failed to transcribe recorded audio. No text extracted.")
                            else:
                                result = process_text(transcription["raw_text"],
                                                      transcription["segmented_conversation"])
                                if result:
                                    process_and_save_result(result)
                                else:
                                    st.error("❌ Failed to process transcribed text.")
                        except Exception as e:
                            st.error(f"❌ Error processing recorded audio: {e}")
                        finally:
                            try:
                                os.unlink(tmp_audio_path)
                            except Exception:
                                pass

        # Fallback to a simpler approach if st.audio_recorder is not available
        except AttributeError:
            st.warning(
                "Your Streamlit version doesn't support audio recording. Please upgrade to Streamlit 1.18.0 or later.")
            st.info("Alternatively, you can use the 'Audio File' option to upload a pre-recorded conversation.")

            # Simple text input fallback
            st.subheader("Alternative: Manual Transcript Entry")
            st.write("If you can't record audio, you can manually enter the conversation transcript here:")

            conversation_text = st.text_area(
                "Enter the doctor-patient conversation:",
                height=300,
                placeholder="Doctor: How are you feeling today?\nPatient: I've been experiencing headaches for the past week...",
                help="Format as 'Doctor: ' and 'Patient: ' to indicate who is speaking."
            )

            if conversation_text and st.button("Process Transcript"):
                try:
                    # Create a simple segmented conversation from the text
                    segmented_conversation = []
                    for line in conversation_text.split("\n"):
                        if line.strip():
                            if line.lower().startswith("doctor:"):
                                speaker = "Doctor"
                                text = line[len("doctor:"):].strip()
                            elif line.lower().startswith("patient:"):
                                speaker = "Patient"
                                text = line[len("patient:"):].strip()
                            else:
                                # Assume the previous speaker continues if no speaker prefix
                                if segmented_conversation:
                                    speaker = segmented_conversation[-1]["speaker"]
                                    text = line.strip()
                                else:
                                    speaker = "Unknown"
                                    text = line.strip()

                            if text:  # Only add if there's actual text content
                                segmented_conversation.append({
                                    "speaker": speaker,
                                    "text": text
                                })

                    result = process_text(conversation_text, segmented_conversation)
                    if result:
                        process_and_save_result(result)
                    else:
                        st.error("❌ Failed to process text.")
                except Exception as e:
                    st.error(f"❌ Error processing text: {e}")


def show_clinical_assistant_page():
    pass


def show_disease_diagnosis_page():
    pass


def process_and_save_result(result):
    if os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []

    existing_ids = [item.get('patient_id') for item in existing_data]
    if result['patient_id'] in existing_ids:
        for i, item in enumerate(existing_data):
            if item['patient_id'] == result['patient_id']:
                existing_data[i] = result
                st.info(f"⚠️ Updated existing record for patient {result['patient_id']}")
    else:
        existing_data.append(result)
        st.success(f"✅ Added new patient record: {result['patient_id']}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(existing_data, f, indent=2)

    os.makedirs(output_directory, exist_ok=True)
    save_patient_records(existing_data, output_directory)

    try:
        collection = get_vector_db()
        add_patient_record(collection, result)
        st.success("✅ Patient record added to vector database")
    except Exception as e:
        st.error(f"❌ Error adding to vector database: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as report_file:
        generate_pdf_report(result, report_file.name)
        with open(report_file.name, "rb") as f:
            st.download_button(
                label="📥 Download Patient Report PDF",
                data=f,
                file_name=f"patient_report_{result['patient_id']}.pdf",
                mime="application/pdf"
            )

    st.subheader("Extracted Patient Data")
    tabs = st.tabs(["Summary", "Full JSON", "Segmented Conversation"])
    with tabs[0]:
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
    with tabs[1]:
        st.json(result)
    with tabs[2]:
        st.subheader("Segmented Conversation")
        for segment in result.get('segmented_conversation', []):
            st.write(f"**{segment['speaker']}:** {segment['text']}")

    st.session_state.refresh_data = True


if __name__ == "__main__":
    main()
