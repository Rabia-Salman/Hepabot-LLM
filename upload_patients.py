import argparse
import json
import glob
import os
import logging
from tqdm import tqdm
from vector_db import initialize_vector_db, clear_vector_db
from data_processor import process_json_file

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def batch_upload_patients(input_path, db_path="./chroma_db"):
    """
    Batch upload patient records from files

    Args:
        input_path (str): Path to JSON file or directory with JSON files
        db_path (str): Path to the ChromaDB database

    Returns:
        int: Number of records processed

    Note: Automatically refreshes the patient browser after processing
    """
    # Initialize the vector database
    collection = initialize_vector_db(db_path)
    if collection is None:
        logger.error("Failed to initialize vector database")
        return 0

    total_processed = 0

    # Check if input is a directory or file
    if os.path.isdir(input_path):
        # Process all JSON files in the directory
        json_files = glob.glob(os.path.join(input_path, "*.json"))
        logger.info(f"Found {len(json_files)} JSON files in directory")

        for json_file in tqdm(json_files, desc="Processing files"):
            records_processed = process_json_file(json_file, collection)
            total_processed += records_processed
            logger.info(f"Processed {records_processed} records from {json_file}")

    elif os.path.isfile(input_path) and input_path.endswith('.json'):
        # Process a single JSON file
        records_processed = process_json_file(input_path, collection)
        total_processed = records_processed
        logger.info(f"Processed {records_processed} records from {input_path}")

    else:
        logger.error(f"Invalid input path: {input_path}")

    # Force refresh of patient browser after processing
    trigger_browser_refresh()

    logger.info(f"Total processed records: {total_processed}")
    return total_processed


def reset_database(db_path="./chroma_db"):
    """
    Delete all records from the vector database

    Args:
        db_path (str): Path to the ChromaDB database

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Clearing all records from database...")
        success = clear_vector_db(db_path)

        # Refresh the patient browser to reflect empty database
        trigger_browser_refresh()

        if success:
            logger.info("Successfully cleared all records from database")
            return True
        else:
            logger.error("Failed to clear records from database")
            return False
    except Exception as e:
        logger.error(f"Error resetting database: {str(e)}")
        return False


def trigger_browser_refresh():
    """
    Trigger a refresh of the patient browser interface
    """
    try:
        # Create a signal file that the browser component watches
        with open("browser_refresh_signal.txt", "w") as f:
            import time
            f.write(str(time.time()))
        logger.info("Sent signal to refresh patient browser")
    except Exception as e:
        logger.error(f"Failed to trigger browser refresh: {str(e)}")


# Function removed as we're using the tab argument approach instead


def main():
    parser = argparse.ArgumentParser(description="Batch upload patient records to vector database")
    parser.add_argument("input", nargs="?", help="Path to JSON file or directory with JSON files")
    parser.add_argument("--db", default="./chroma_db", help="Path to ChromaDB database")
    parser.add_argument("--reset", action="store_true", help="Reset the database before uploading")
    parser.add_argument("--tab", type=int, help="Select tab functionality: 5 for database reset")
    args = parser.parse_args()

    # Handle tab-specific functionality
    if args.tab == 5:
        # This is the "reset database" tab
        success = reset_database(args.db)
        if success:
            print("Database reset successful (Tab 5)")
        else:
            print("Failed to reset database")
        return

    # If no input provided for regular operations, show error
    if not args.input and args.tab is None:
        parser.error("the following arguments are required: input")
        return

    # Reset database if requested via --reset flag
    if args.reset:
        success = reset_database(args.db)
        if success:
            print("Database reset successful")
        else:
            print("Database reset failed")
            return

    # Only process files if input is provided
    if args.input:
        records_processed = batch_upload_patients(args.input, args.db)
        print(f"Total records processed: {records_processed}")


if __name__ == "__main__":
    main()