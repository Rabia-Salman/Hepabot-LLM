import chromadb
from chromadb.utils import embedding_functions
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def initialize_vector_db(db_path="./chroma_db_patients"):
    """Initialize or create a vector database."""
    try:
        client = chromadb.PersistentClient(db_path)
        default_ef = embedding_functions.DefaultEmbeddingFunction()

        try:
            collection = client.get_collection("medical_records", embedding_function=default_ef)
            logger.info("Connected to existing medical records database.")
        except Exception:
            collection = client.create_collection(
                "medical_records",
                embedding_function=default_ef,
                metadata={"description": "Medical patient records with structured data"}
            )
            logger.info("Created new medical records database.")

        return collection
    except Exception as e:
        logger.error(f"Error initializing vector database: {e}")
        return None

def _safe_join(items):
    """Safely join list of strings. Ignore if not a list."""
    if isinstance(items, list):
        return ', '.join(map(str, items))
    return str(items)

def _safe_lab_results(lab_results):
    """Flatten lab results for string representation."""
    lines = []
    if isinstance(lab_results, list):
        for idx, item in enumerate(lab_results):
            if isinstance(item, dict):
                for k, v in item.items():
                    lines.append(f"{k}: {v}")
            else:
                lines.append(str(item))
    elif isinstance(lab_results, dict):
        for k, v in lab_results.items():
            lines.append(f"{k}: {v}")
    else:
        lines.append(str(lab_results))
    return "\n".join(lines)

def add_patient_record(collection, patient_data):
    """Add a patient record to the vector database."""
    if collection is None:
        logger.warning("Vector database not initialized.")
        return False

    try:
        patient_id = patient_data.get('patient_id', str(datetime.now().timestamp()))
        structured_data = patient_data.get('structured_data', {})
        raw_text = patient_data.get('raw_text', '')

        demographics = structured_data.get('PatientDemographics', {})
        metadata = {
            "patient_id": patient_id,
            "gender": demographics.get('Gender', ''),
            "age": demographics.get('Age', ''),
            "mrn": demographics.get('MRN', ''),
            "diagnosis": demographics.get('Diagnosis', ''),
            "document_type": "medical_record",
            "processed_date": datetime.now().isoformat()
        }

        # Build text chunks
        chunks = []
        chunk_ids = []

        def add_chunk(title, content):
            if content.strip():
                chunks.append(f"{title}:\n{content.strip()}")
                chunk_ids.append(f"{patient_id}_{title.lower().replace(' ', '_')}")

        add_chunk("PATIENT DEMOGRAPHICS", f"""
        ID: {patient_id}
        Gender: {demographics.get('Gender', '')}
        Age: {demographics.get('Age', '')}
        MRN: {demographics.get('MRN', '')}
        Diagnosis: {demographics.get('Diagnosis', '')}
        """)

        clinical_summary = structured_data.get('ClinicalSummary', {})
        add_chunk("CLINICAL SUMMARY", f"""
        Active Symptoms: {_safe_join(clinical_summary.get('ActiveSymptoms', []))}
        Negative Findings: {_safe_join(clinical_summary.get('NegativeFindings', []))}
        """)

        add_chunk("DIAGNOSTIC CONCLUSIONS", _safe_join(structured_data.get('DiagnosticConclusions', [])))

        therapeutic = structured_data.get('TherapeuticInterventions', {})
        add_chunk("THERAPEUTIC INTERVENTIONS", f"""
        Medications: {_safe_join(therapeutic.get('Medications', []))}
        Procedures: {_safe_join(therapeutic.get('Procedures', []))}
        """)

        diagnostic_evidence = structured_data.get('DiagnosticEvidence', {})
        add_chunk("DIAGNOSTIC EVIDENCE", f"""
        Lab Results:
        {_safe_lab_results(diagnostic_evidence.get('LabResults', []))}

        Imaging Findings: {_safe_join(diagnostic_evidence.get('ImagingFindings', []))}
        """)

        chronic_conditions = structured_data.get('ChronicConditions', {})
        add_chunk("CHRONIC CONDITIONS", f"""
        Chronic Diseases: {_safe_join(chronic_conditions.get('ChronicDiseases', []))}
        Comorbidities: {_safe_join(chronic_conditions.get('Comorbidities', []))}
        """)

        follow_up = structured_data.get('Follow-upPlan', {})
        add_chunk("FOLLOW-UP PLAN", f"""
        Planned Consultations: {_safe_join(follow_up.get('PlannedConsultations', []))}
        Scheduled Tests: {_safe_join(follow_up.get('ScheduledTests', []))}
        Next Appointments: {_safe_join(follow_up.get('NextAppointmentDetails', []))}
        """)

        add_chunk("VISIT TIMELINE", _safe_join(structured_data.get('VisitTimeline', [])))

        summary = structured_data.get('SummaryNarrative', {})
        add_chunk("SUMMARY NARRATIVE", f"""
        Clinical Course: {summary.get('ClinicalCourseProgression', '')}
        Diagnostic Journey: {summary.get('DiagnosticJourney', '')}
        Treatment Response: {summary.get('TreatmentResponse', '')}
        Ongoing Concerns: {summary.get('OngoingConcerns', '')}
        """)

        # Raw text chunks
        if raw_text:
            max_chunk_size = 1000
            for i in range(0, len(raw_text), max_chunk_size):
                chunk = raw_text[i:i+max_chunk_size]
                add_chunk(f"RAW TEXT PART {i//max_chunk_size + 1}", chunk)

        # Validate: ensure all documents are strings
        chunks = [str(chunk) for chunk in chunks]

        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=[metadata] * len(chunks)
        )

        logger.info(f"Successfully added patient {patient_id} to vector database")
        return True

    except Exception as e:
        logger.error(f"Error adding patient {patient_id} to vector database: {e}")
        return False

def search_records(collection, query, filter_metadata=None, limit=5):
    """Search the vector database."""
    if collection is None:
        logger.warning("Vector database not initialized.")
        return None

    try:
        return collection.query(
            query_texts=[query],
            n_results=limit,
            where=filter_metadata or {}
        )
    except Exception as e:
        logger.error(f"Error searching vector database: {e}")
        return None


#
# import chromadb
# from chromadb.utils import embedding_functions
# from datetime import datetime
# import logging
#
# # Setup logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
#
# def initialize_vector_db(db_path="./chroma_db_patients"):
#     """Initialize or create a vector database."""
#     try:
#         client = chromadb.PersistentClient(db_path)
#         default_ef = embedding_functions.DefaultEmbeddingFunction()
#
#         try:
#             collection = client.get_collection("medical_records", embedding_function=default_ef)
#             logger.info("Connected to existing medical records database.")
#         except Exception:
#             collection = client.create_collection(
#                 "medical_records",
#                 embedding_function=default_ef,
#                 metadata={"description": "Medical patient records with structured data"}
#             )
#             logger.info("Created new medical records database.")
#
#         return collection
#     except Exception as e:
#         logger.error(f"Error initializing vector database: {e}")
#         return None
#
# def _safe_join(items):
#     """Safely join list of strings. Ignore if not a list."""
#     if isinstance(items, list):
#         return ', '.join(map(str, items))
#     return str(items)
#
# def _safe_lab_results(lab_results):
#     """Flatten lab results for string representation."""
#     lines = []
#     if isinstance(lab_results, list):
#         for idx, item in enumerate(lab_results):
#             if isinstance(item, dict):
#                 for k, v in item.items():
#                     lines.append(f"{k}: {v}")
#             else:
#                 lines.append(str(item))
#     elif isinstance(lab_results, dict):
#         for k, v in lab_results.items():
#             lines.append(f"{k}: {v}")
#     else:
#         lines.append(str(lab_results))
#     return "\n".join(lines)
#
# def add_patient_record(collection, patient_data):
#     """Add a patient record to the vector database."""
#     if collection is None:
#         logger.warning("Vector database not initialized.")
#         return False
#
#     try:
#         patient_id = patient_data.get('patient_id', str(datetime.now().timestamp()))
#         structured_data = patient_data.get('structured_data', {})
#         raw_text = patient_data.get('raw_text', '')
#
#         demographics = structured_data.get('PatientDemographics', {})
#         metadata = {
#             "patient_id": patient_id,
#             "gender": demographics.get('Gender', ''),
#             "age": demographics.get('Age', ''),
#             "mrn": demographics.get('MRN', ''),
#             "diagnosis": demographics.get('Diagnosis', ''),
#             "document_type": "medical_record",
#             "processed_date": datetime.now().isoformat()
#         }
#
#         # Build text chunks
#         chunks = []
#         chunk_ids = []
#
#         def add_chunk(title, content):
#             if content.strip():
#                 chunks.append(f"{title}:\n{content.strip()}")
#                 chunk_ids.append(f"{patient_id}_{title.lower().replace(' ', '_')}")
#
#         add_chunk("PATIENT DEMOGRAPHICS", f"""
#         ID: {patient_id}
#         Gender: {demographics.get('Gender', '')}
#         Age: {demographics.get('Age', '')}
#         MRN: {demographics.get('MRN', '')}
#         Diagnosis: {demographics.get('Diagnosis', '')}
#         """)
#
#         clinical_summary = structured_data.get('ClinicalSummary', {})
#         add_chunk("CLINICAL SUMMARY", f"""
#         Active Symptoms: {_safe_join(clinical_summary.get('ActiveSymptoms', []))}
#         Negative Findings: {_safe_join(clinical_summary.get('NegativeFindings', []))}
#         """)
#
#         add_chunk("DIAGNOSTIC CONCLUSIONS", _safe_join(structured_data.get('DiagnosticConclusions', [])))
#
#         therapeutic = structured_data.get('TherapeuticInterventions', {})
#         add_chunk("THERAPEUTIC INTERVENTIONS", f"""
#         Medications: {_safe_join(therapeutic.get('Medications', []))}
#         Procedures: {_safe_join(therapeutic.get('Procedures', []))}
#         """)
#
#         diagnostic_evidence = structured_data.get('DiagnosticEvidence', {})
#         add_chunk("DIAGNOSTIC EVIDENCE", f"""
#         Lab Results:
#         {_safe_lab_results(diagnostic_evidence.get('LabResults', []))}
#
#         Imaging Findings: {_safe_join(diagnostic_evidence.get('ImagingFindings', []))}
#         """)
#
#         chronic_conditions = structured_data.get('ChronicConditions', {})
#         add_chunk("CHRONIC CONDITIONS", f"""
#         Chronic Diseases: {_safe_join(chronic_conditions.get('ChronicDiseases', []))}
#         Comorbidities: {_safe_join(chronic_conditions.get('Comorbidities', []))}
#         """)
#
#         follow_up = structured_data.get('Follow-upPlan', {})
#         add_chunk("FOLLOW-UP PLAN", f"""
#         Planned Consultations: {_safe_join(follow_up.get('PlannedConsultations', []))}
#         Scheduled Tests: {_safe_join(follow_up.get('ScheduledTests', []))}
#         Next Appointments: {_safe_join(follow_up.get('NextAppointmentDetails', []))}
#         """)
#
#         add_chunk("VISIT TIMELINE", _safe_join(structured_data.get('VisitTimeline', [])))
#
#         summary = structured_data.get('SummaryNarrative', {})
#         add_chunk("SUMMARY NARRATIVE", f"""
#         Clinical Course: {summary.get('ClinicalCourseProgression', '')}
#         Diagnostic Journey: {summary.get('DiagnosticJourney', '')}
#         Treatment Response: {summary.get('TreatmentResponse', '')}
#         Ongoing Concerns: {summary.get('OngoingConcerns', '')}
#         """)
#
#         # Raw text chunks
#         if raw_text:
#             max_chunk_size = 1000
#             for i in range(0, len(raw_text), max_chunk_size):
#                 chunk = raw_text[i:i+max_chunk_size]
#                 add_chunk(f"RAW TEXT PART {i//max_chunk_size + 1}", chunk)
#
#         # Validate: ensure all documents are strings
#         chunks = [str(chunk) for chunk in chunks]
#
#         collection.add(
#             documents=chunks,
#             ids=chunk_ids,
#             metadatas=[metadata] * len(chunks)
#         )
#
#         logger.info(f"Successfully added patient {patient_id} to vector database")
#         return True
#
#     except Exception as e:
#         logger.error(f"Error adding patient {patient_id} to vector database: {e}")
#         return False
#
# def search_records(collection, query, filter_metadata=None, limit=5):
#     """Search the vector database."""
#     if collection is None:
#         logger.warning("Vector database not initialized.")
#         return None
#
#     try:
#         return collection.query(
#             query_texts=[query],
#             n_results=limit,
#             where=filter_metadata or {}
#         )
#     except Exception as e:
#         logger.error(f"Error searching vector database: {e}")
#         return None
#
# def delete_patient_record(collection, patient_id):
#     """Delete all records associated with a patient ID from the vector database."""
#     if collection is None:
#         logger.warning("Vector database not initialized.")
#         return False
#
#     try:
#         # Delete all chunks where metadata patient_id matches
#         collection.delete(where={"patient_id": patient_id})
#         logger.info(f"Successfully deleted patient {patient_id} from vector database")
#         return True
#     except Exception as e:
#         logger.error(f"Error deleting patient {patient_id} from vector database: {e}")
#         return False
