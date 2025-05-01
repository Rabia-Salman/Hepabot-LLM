import streamlit as st
import json
import re
from typing import Dict, Any
from pdfminer.high_level import extract_text
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def load_conversations_from_pdf(pdf_file) -> str:
    """Extract raw text from uploaded PDF file"""
    try:
        return extract_text(pdf_file)
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""


def extract_metadata(raw_text: str) -> Dict[str, Any]:
    """Extract structured metadata from text using LLM"""
    # You can switch between models as needed
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        # openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Improved prompt to ensure valid JSON output
    extraction_template = ChatPromptTemplate.from_template("""
    You are a medical information extraction specialist. Your task is to extract structured information from medical text.

    MEDICAL TEXT:
    ```
    {conversation}
    ```

    INSTRUCTIONS:
    1. Extract key medical information from the text
    2. Format your response as a VALID JSON object with the structure shown below
    3. ONLY return the JSON object - no additional text, explanations, or markdown
    4. Ensure all JSON keys and values are properly quoted
    5. Use None as default values. don't leave any string to be empty
    6. infer gender from conversation only if 100% confident
    7. don't output fields for which no data is given instead at the end of chat suggest doctor to 
    ask for that data in a very professional and brief way
    

    OUTPUT FORMAT:
    ```json
    {{
      "PatientDemographics": {{
        "Gender": "",
        "Age": "",
        "MRN": "",
        "Diagnosis": ""
      }},
      "ClinicalSummary": {{
        "ActiveSymptoms": [],
        "NegativeFindings": []
      }},
      "DiagnosticConclusions": [],
      "TherapeuticInterventions": {{
        "Medications": [],
        "Procedures": []
      }},
      "DiagnosticEvidence": {{
        "ImagingFindings": [],
        "LabResults": [],
        "PathologyFindings": []
      }},
      "ChronicConditions": {{
        "ChronicDiseases": [],
        "Comorbidities": []
      }},
      "Follow-upPlan": {{
        "PlannedConsultations": [],
        "ScheduledTests": [],
        "NextAppointmentDetails": []
      }},
      "VisitTimeline": [],
      "SummaryNarrative": {{
        "ClinicalCourseProgression": "",
        "DiagnosticJourney": "",
        "TreatmentResponse": "",
        "OngoingConcerns": ""
      }}
    }}
    ```

    REMINDER: Return ONLY the JSON object with no additional text.
    """)

    chain = extraction_template | llm | StrOutputParser()

    try:
        # Get raw result from LLM
        with st.spinner("Extracting structured data with AI..."):
            result = chain.invoke({"conversation": raw_text})

        # Clean up the result - remove markdown code blocks if present
        result = result.strip()
        if result.startswith("```json"):
            result = result.replace("```json", "", 1)
        if result.startswith("```"):
            result = result.replace("```", "", 1)
        if result.endswith("```"):
            result = result[:-3]

        result = result.strip()

        # Parse the string result into a JSON object
        return json.loads(result)
    except json.JSONDecodeError as e:
        st.error(f"JSON parsing error: {e}")

        # Attempt basic recovery
        try:
            # Find anything that looks like JSON
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, result)
            if match:
                potential_json = match.group(0)
                st.warning("Attempting to parse extracted JSON pattern...")
                return json.loads(potential_json)
        except:
            pass

        # Return a default structure on error
        return get_default_structure()
    except Exception as e:
        st.error(f"Error extracting metadata: {e}")
        return get_default_structure()


def get_default_structure() -> Dict[str, Any]:
    """Return default structure for when extraction fails"""
    return {
        "PatientDemographics": {"Gender": "", "Age": "", "MRN": "", "Diagnosis": ""},
        "ClinicalSummary": {"ActiveSymptoms": [], "NegativeFindings": []},
        "DiagnosticConclusions": [],
        "TherapeuticInterventions": {"Medications": [], "Procedures": []},
        "DiagnosticEvidence": {"ImagingFindings": [], "LabResults": [], "PathologyFindings": []},
        "ChronicConditions": {"ChronicDiseases": [], "Comorbidities": []},
        "Follow-upPlan": {"PlannedConsultations": [], "ScheduledTests": [], "NextAppointmentDetails": []},
        "VisitTimeline": [],
        "SummaryNarrative": {
            "ClinicalCourseProgression": "",
            "DiagnosticJourney": "",
            "TreatmentResponse": "",
            "OngoingConcerns": ""
        }
    }


def extract_basic_metadata(raw_text: str) -> Dict[str, Any]:
    """Extract basic metadata using regex patterns - fallback method"""
    metadata = get_default_structure()

    # Basic regex patterns for common medical information
    patterns = {
        "gender": r"(?:gender|sex):\s*(male|female|other)",
        "age": r"(?:age|years old):\s*(\d+)",
        "mrn": r"(?:mrn|medical record number|record number):\s*(\d+)",
        "diagnosis": r"(?:diagnosis|impression|assessment):\s*([^\n\.]+)",
        "symptoms": r"(?:symptoms|complaints|presenting with):\s*([^\n\.]+)",
        "medications": r"(?:medications|meds|prescriptions):\s*([^\n\.]+)",
    }

    # Extract using regex
    for field, pattern in patterns.items():
        matches = re.finditer(pattern, raw_text, re.IGNORECASE)
        extracted = [match.group(1).strip() for match in matches]

        if not extracted:
            continue

        # Map to our structure
        if field == "gender" and extracted:
            metadata["PatientDemographics"]["Gender"] = extracted[0]
        elif field == "age" and extracted:
            metadata["PatientDemographics"]["Age"] = extracted[0]
        elif field == "mrn" and extracted:
            metadata["PatientDemographics"]["MRN"] = extracted[0]
        elif field == "diagnosis" and extracted:
            metadata["PatientDemographics"]["Diagnosis"] = extracted[0]
        elif field == "symptoms" and extracted:
            metadata["ClinicalSummary"]["ActiveSymptoms"] = extracted
        elif field == "medications" and extracted:
            metadata["TherapeuticInterventions"]["Medications"] = extracted

    return metadata


def main():
    st.set_page_config(page_title="Clinical Summary",
                       page_icon="🏥",
                       layout="wide")

    st.title("Clinical Summary")
    st.write("Upload a medical PDF document to generate summary report please.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")


    if uploaded_file is not None:
        # Display upload success
        st.success(f"File uploaded: {uploaded_file.name}")

        # Extract text from PDF
        with st.spinner("Extracting text from PDF..."):
            raw_text = load_conversations_from_pdf(uploaded_file)

        # Skip empty files
        if not raw_text:
            st.error("No text could be extracted from the PDF.")
        else:
            # Extract structured metadata
            structured_data = extract_metadata(raw_text)

            # Create result object
            result = {
                "filename": uploaded_file.name,
                "raw_text": raw_text,
                "structured_data": structured_data
            }

            # Display results
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Extracted Raw Text")
                st.text_area("", raw_text[:1000] + ("..." if len(raw_text) > 1000 else ""),
                             height=300)

            with col2:
                st.subheader("Structured Data")
                st.json(structured_data)

            # Download buttons
            st.subheader("Download Results")
            col1, col2 = st.columns(2)

            with col1:
                json_data = json.dumps(result, indent=2)
                st.download_button(
                    label="Download Full Results (JSON)",
                    data=json_data,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_extracted.json",
                    mime="application/json"
                )

            with col2:
                structured_json = json.dumps(structured_data, indent=2)
                st.download_button(
                    label="Download Structured Data Only (JSON)",
                    data=structured_json,
                    file_name=f"{uploaded_file.name.replace('.pdf', '')}_structured.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()