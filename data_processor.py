import json
import re
import logging
import os
from vector_db import add_patient_record

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_json_string(json_string):
    """Parse a JSON string that might contain multiple patient records

    Args:
        json_string (str): The JSON string to parse

    Returns:
        list: List of parsed patient records
    """
    try:
        # First try to parse as a regular JSON
        data = json.loads(json_string)

        # If it's a dict, wrap it in a list for consistent handling
        if isinstance(data, dict):
            return [data]
        return data

    except json.JSONDecodeError:
        # If that fails, try to extract valid JSON objects
        patient_records = []

        # Find JSON objects using regex pattern matching
        pattern = r'({[^{}]*(?:{[^{}]*}[^{}]*)*})'
        matches = re.findall(pattern, json_string)

        for match in matches:
            try:
                record = json.loads(match)
                if record and isinstance(record, dict):
                    patient_records.append(record)
            except json.JSONDecodeError:
                continue

        return patient_records


def process_patient_json(json_data, collection):
    """Process patient data from JSON

    Args:
        json_data (str): JSON string or dict containing patient data
        collection: Vector database collection

    Returns:
        int: Number of records processed
    """
    if not json_data:
        logger.error("No JSON data provided")
        return 0

    # Parse the JSON if it's a string
    data = parse_json_string(json_data) if isinstance(json_data, str) else json_data

    if not data:
        logger.error("No valid patient data found")
        return 0

    processed_count = 0

    # Process based on data structure
    if isinstance(data, list):
        # List of patient records
        for patient_record in data:
            # Check if this looks like a patient record
            if isinstance(patient_record, dict) and any(
                    key in patient_record for key in ['patient_id', 'raw_text', 'structured_data']):
                if add_patient_record(collection, patient_record):
                    processed_count += 1

    elif isinstance(data, dict):
        # Single patient record or container
        if any(key in data for key in ['patient_id', 'raw_text', 'structured_data']):
            # Single patient record
            if add_patient_record(collection, data):
                processed_count = 1
        else:
            # Container with multiple records
            for key, value in data.items():
                if isinstance(value, dict) and any(k in value for k in ['patient_id', 'raw_text', 'structured_data']):
                    # Ensure patient_id is included
                    if 'patient_id' not in value:
                        value['patient_id'] = key
                    if add_patient_record(collection, value):
                        processed_count += 1
                elif isinstance(value, list):
                    # List of patients under a key
                    for patient in value:
                        if isinstance(patient, dict) and add_patient_record(collection, patient):
                            processed_count += 1

    logger.info(f"Processed {processed_count} patient records")
    return processed_count


def process_json_file(file_path, collection):
    """Process a JSON file containing patient records

    Args:
        file_path (str): Path to the JSON file
        collection: Vector database collection

    Returns:
        int: Number of records processed
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return 0

    try:
        with open(file_path, 'r') as f:
            content = f.read()

        return process_patient_json(content, collection)

    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return 0