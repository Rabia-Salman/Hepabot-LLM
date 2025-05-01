import json
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_patient_records(data, output_dir):
    """
    Saves each patient record in the data list to a separate JSON file.

    Args:
        data (list or str): Either a list of patient records or a path to the JSON file
        output_dir (str): Directory where individual JSON files will be saved
    """
    os.makedirs(output_dir, exist_ok=True)

    # Handle different input types
    if isinstance(data, str):
        # It's a file path
        try:
            with open(data, 'r', encoding='utf-8') as f:
                patient_records = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load patient data from {data}: {e}")
            return False
    elif isinstance(data, list):
        # It's already a list of records
        patient_records = data
    else:
        logger.error(f"Expected a list of patient records or a file path, got {type(data)}")
        return False

    if not isinstance(patient_records, list):
        logger.error(f"Expected a list of patient records at top level, got {type(patient_records)}")
        return False

    count = 0
    for idx, patient in enumerate(patient_records):
        # Try to use patient_id; fallback to index
        patient_id = patient.get("patient_id", f"patient_{idx}")

        # Create a safe filename by replacing invalid characters
        safe_patient_id = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in str(patient_id))
        output_path = os.path.join(output_dir, f"{safe_patient_id}.json")

        try:
            with open(output_path, 'w', encoding='utf-8') as out_f:
                json.dump(patient, out_f, ensure_ascii=False, indent=4)
            count += 1
        except Exception as e:
            logger.error(f"Failed to save patient {patient_id}: {e}")

    logger.info(f"✅ Saved {count} patient files into '{output_dir}'")
    return True


if __name__ == "__main__":
    # Adjust paths if needed
    input_json_path = "doctor_patient_data_80.json"
    output_directory = "split_patient_files"

    save_patient_records(input_json_path, output_directory)