import os
import pandas as pd
from fpdf import FPDF
import unicodedata


# Function to clean and normalize text
def clean_text(text):
    """
    Replace unsupported characters with their closest equivalents or remove them.
    """
    text = unicodedata.normalize("NFKD", text)  # Normalize text (e.g., convert fancy quotes to plain ones)
    return text.encode("latin-1", errors="ignore").decode(
        "latin-1")  # Remove characters that can't be encoded in latin-1


# Load the Excel file and sheet
file_path = "data/Data_1.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")

# Get the last column
conversation_notes = df.iloc[:, -1]
print(conversation_notes.head())

# Directory to save the PDF files
output_dir = "/ollama-fundamentals/history_physical_pdfs"
os.makedirs(output_dir, exist_ok=True)


# PDF generation class
class PDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    def add_text(self, text):
        self.multi_cell(0, 10, text)


# Save each row as a separate PDF
file_paths = []
for idx, text in enumerate(conversation_notes.dropna(), start=1):  # Drop NaN values
    pdf = PDF()
    cleaned_text = clean_text(str(text))  # Clean the text before adding it to the PDF
    pdf.add_text(cleaned_text)

    file_name = f"history_physical_{idx}.pdf"
    file_path = os.path.join(output_dir, file_name)
    pdf.output(file_path)
    file_paths.append(file_path)

# Show first 5 file paths
print(file_paths[:5])
