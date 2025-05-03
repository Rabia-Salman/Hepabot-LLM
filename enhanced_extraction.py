# enhanced_extraction.py

import re
import os
import json
from typing import Dict, Any, List
from pdfminer.high_level import extract_text
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


def load_conversations_from_pdf(pdf_path: str) -> str:
    """Extract raw text from PDF file"""
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""


def extract_metadata(raw_text: str) -> Dict[str, Any]:
    """Extract structured metadata from text using LLM"""
    # llm = ChatOllama(model="llama3.2", temperature=0)  # Lower temperature for more deterministic outputs
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # llm = ChatOpenAI(
    #     model="gpt-4-turbo",
    #     temperature=0,
    #     api_key="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"
    # )

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
    5. Use None as default values e.g ["None"]. don't leave any string to be empty
    6. Diagnosis can only be one of the following: Budd-Chiari syndrome, Cholangiocarcinoma ,Chronic viral hepatitis C ,
     Hepatic fibrosis ,Hepatocellular Carcinoma ,hepatitis C ,Chronic hepatic failure. if found anything else that may be a spelling mistake
     try to correct it to one of the these based on what it seems like else assign "None"
    7. if diagnosis is in lower case or in any style , format it exactly as in point 6    
    8. format gender always like "Male" or "Female"
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
        result = chain.invoke({"conversation": raw_text})

        # Debug - print a sample of what we got
        print(f"LLM output sample (first 500 chars): {result[:500]}...")

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
        print(f"JSON parsing error: {e}")
        print(f"Raw LLM output: {result}")

        # Attempt basic recovery
        try:
            # Find anything that looks like JSON
            json_pattern = r'\{[\s\S]*\}'
            match = re.search(json_pattern, result)
            if match:
                potential_json = match.group(0)
                print(f"Attempting to parse extracted JSON pattern...")
                return json.loads(potential_json)
        except:
            pass

        # Return a default structure on error
        return get_default_structure()
    except Exception as e:
        print(f"Error extracting metadata: {e}")
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


def process_pdf(pdf_path: str) -> Dict[str, Any]:
    """Process a single PDF file to extract and structure data"""
    # Extract text from PDF
    raw_text = load_conversations_from_pdf(pdf_path)

    # Skip empty files
    if not raw_text:
        print(f"Warning: No text extracted from {pdf_path}")
        return None

    # Print first 100 chars of text for debugging
    print(f"Raw text sample: {raw_text[:100]}...")

    # Extract structured metadata
    structured_data = extract_metadata(raw_text)

    return {
        "patient_id": os.path.basename(pdf_path).replace(".pdf", ""),
        "raw_text": raw_text,
        "structured_data": structured_data
    }


def process_all_pdfs(pdf_dir: str) -> List[Dict[str, Any]]:
    """Process all PDFs in directory and return structured data"""
    all_data = []

    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

    for pdf_file in pdf_files:
        full_path = os.path.join(pdf_dir, pdf_file)
        print(f"\nProcessing: {pdf_file}")

        # Process the file
        result = process_pdf(full_path)
        if result:
            all_data.append(result)

    return all_data


# Alternative method using a simple rule-based extraction
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
            for med in extracted:
                metadata["TherapeuticInterventions"]["Medications"].append({"Medication": med})

    return metadata


if __name__ == "__main__":
    # For testing
    PDF_DIR = "/Users/aks/Documents/ollama-models-llm/ollama-fundamentals/ollama-fundamentals/history_physical_pdfs"
    OUTPUT_JSON = "doctor_patient_data_80.json"

    data = process_all_pdfs(PDF_DIR)

    # Save to JSON
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved structured data for {len(data)} files to {OUTPUT_JSON}")
