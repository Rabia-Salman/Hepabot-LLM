# enhanced_extraction.py
import os
import json
from tqdm import tqdm
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from utils import load_conversations_from_pdf
import re
from pdfminer.high_level import extract_text

# Initialize LLM
llm = ChatOllama(model="llama3.2", temperature=0.3)  # Lower temperature for more factual responses

# Template for extracting structured information
extraction_template = ChatPromptTemplate.from_template("""
You are a clinical documentation specialist preparing structured medical summaries. Analyze the conversation and extract:

Patient Demographics:
- Gender: {gender}
- Age: {age}
- MRN: {mrn}
- Diagnosis: {diagnosis}

Clinical Summary: [Strictly follow these headings in order]

Active Symptoms: <Presenting symptoms with duration and characteristics>
<Format: Symptom (duration): description>

Negative Findings: <Explicitly denied symptoms/negative review of systems>

Diagnostic Conclusions: <Final/working diagnoses with ICD codes if known>

Therapeutic Interventions: <Medications (dose, frequency, duration)>
<Procedures/surgical plans with dates if mentioned>

Diagnostic Evidence: <Imaging findings with dates>
<Lab results with dates and reference ranges>
<Pathology findings>

Chronic Conditions: <Chronic diseases and comorbidities>

Follow-up Plan: <Planned consultations>
<Scheduled tests with dates>
<Next appointment details>

Visit Timeline: <Chronological list of key events in MM/YYYY format>

Summary Narrative: <3-4 paragraph concise summary highlighting:
- Clinical course progression
- Diagnostic journey
- Treatment response
- Ongoing concerns>

Strict Rules:
1. Preserve exact numerical values and medical terms
2. Maintain temporal sequence of events
3. Include both positive and negative findings
4. Use only the following headings exactly as written
5. Never invent information not explicitly stated
6. Format dates consistently as DD/MM/YYYY
7. Keep paragraphs under 5 lines
8- DO NOT USE ANY DECORATIVE IMAGES AND DO NOT INVENT ANYTHING NOT EXPLICITLY STATED IN THE CONVERSATION.
9- GIVE NEAT AND CLEAN PDF WITHOUT MARKDOWN 
Conversation Data: {conversation}
""")


def extract_metadata(pdf_path):
    """Extract gender, age, MRN from PDF text"""
    text = extract_text(pdf_path)

    metadata = {
        "gender": None,
        "age": None,
        "mrn": None,
        "diagnosis": None
    }

    # Extract using regex patterns
    patterns = {
        "gender": r"Gender:\s*(Male|Female|Other)",
        "age": r"Age:\s*(\d+)",
        "mrn": r"MRN:\s*(\d+)",
        "diagnosis": r"Diagnosis:\s*(.+?)\n"
    }

    for field, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            metadata[field] = match.group(1).strip()

    return metadata


def process_pdf(pdf_path):
    """Process a single PDF and return structured data"""
    # Extract metadata first
    metadata = extract_metadata(pdf_path)

    # Then load conversation turns
    turns = load_conversations_from_pdf(pdf_path)
    conversation_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])

    chain = extraction_template | llm | StrOutputParser()

    # Get the extracted clinical summary
    extracted_text = chain.invoke({
        "gender": metadata["gender"],
        "age": metadata["age"],
        "mrn": metadata["mrn"],
        "diagnosis": metadata["diagnosis"],
        "conversation": conversation_text
    })

    # Add file path for reference
    result = {
        "source_file": os.path.basename(pdf_path),
        "raw_text": extracted_text,
        "metadata": metadata
    }

    return result


def process_all_pdfs(directory):
    """Process all PDFs in the given directory"""
    results = []

    # Get all PDF files in the directory
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

    # Process each PDF file with a progress bar
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        try:
            result = process_pdf(pdf_file)
            results.append(result)
            print(f"Processed: {pdf_file}")
        except Exception as e:
            print(f"Error processing {pdf_file}: {str(e)}")

    return results


def save_results(results, output_file="extracted_medical_data.json"):
    """Save the processed results to a JSON file"""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Directory containing the PDF files
    pdf_directory = "../history_physical_pdfs"

    # Process all PDFs
    results = process_all_pdfs(pdf_directory)

    # Save the results
    save_results(results)

    print(f"Processed {len(results)} PDF files")