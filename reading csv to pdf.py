import os
import pandas as pd
from fpdf import FPDF
import unicodedata


def clean_text(text):
    """Normalize and clean text to remove unsupported characters."""
    text = unicodedata.normalize("NFKD", str(text))
    return text.encode("latin-1", "ignore").decode("latin-1")


# Load the Excel file and select the required columns
file_path = "data/Data_1.xlsx"
df = pd.read_excel(file_path, sheet_name="Sheet1")
selected_columns = df.iloc[:,
                   [0, 1, 3, 11, 13]]  # Columns: Gender, Age, MRN, Outpatient Diagnosis, History and Physical

# Ensure the output directory exists
output_dir = "./ollama-fundamentals/history_physical_pdfs"
os.makedirs(output_dir, exist_ok=True)


class PDF(FPDF):
    """Custom PDF class with consistent formatting."""

    def __init__(self):
        super().__init__()
        self.set_auto_page_break(auto=True, margin=15)
        self.add_page()
        self.set_font("Arial", size=12)

    def add_text(self, text):
        """Add multi-line text to the PDF."""
        self.multi_cell(0, 10, text)


# Generate a PDF for each row with non-missing values
file_paths = []
cleaned_data = selected_columns.dropna()  # Skip rows with any missing data

for count, (_, row) in enumerate(cleaned_data.iterrows(), start=1):
    # Format the data with labels
    formatted_content = (
        f"Gender: {clean_text(row['Gender'])}\n"
        f"Age: {clean_text(row['Age'])}\n"
        f"MRN: {clean_text(row['MRN'])}\n"
        f"Outpatient Diagnosis: {clean_text(row['Outpatient Diagnosis'])}\n"
        f"History and Physical:\n{clean_text(row['History and Physical'])}"
    )

    # Create and save the PDF
    pdf = PDF()
    pdf.add_text(formatted_content)
    filename = os.path.join(output_dir, f"history_physical_{count}.pdf")
    pdf.output(filename)
    file_paths.append(filename)

print(f"Successfully generated {len(file_paths)} PDFs.")
print("Sample paths:", file_paths[:5])
