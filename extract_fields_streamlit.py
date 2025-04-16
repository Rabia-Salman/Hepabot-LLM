import streamlit as st
from extract_fields import extraction_template, llm
from langchain_core.output_parsers import StrOutputParser
from utils import load_conversations_from_pdf
import tempfile
import json
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pathlib import Path

st.set_page_config(page_title="Medical Summary", layout="centered")
st.title("🩺 Patient Details")

uploaded_file = st.file_uploader("Upload a clinical conversation PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    with st.spinner("🔍 Analyzing data ..."):
        try:
            turns = load_conversations_from_pdf(tmp_path)
            conversation_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])
            chain = extraction_template | llm | StrOutputParser()
            raw_output = chain.invoke({"conversation": conversation_text})

            # Clean and save output in session state
            def clean_output(text):
                cleaned = re.sub(r"^```(?:json)?\s*|```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
                return cleaned.strip()

            cleaned_output = clean_output(raw_output)
            st.session_state["raw_output"] = raw_output
            st.session_state["cleaned_output"] = cleaned_output
            pdf_path = Path(tempfile.gettempdir()) / "medical_summary.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            width, height = letter

            lines = st.session_state.raw_output.splitlines()
            max_chars_per_line = 100
            formatted_lines = []
            for line in lines:
                if len(line) <= max_chars_per_line:
                    formatted_lines.append(line)
                else:
                    while len(line) > max_chars_per_line:
                        formatted_lines.append(line[:max_chars_per_line])
                        line = line[max_chars_per_line:]
                    if line:
                        formatted_lines.append(line)

            y = height - 40
            for line in formatted_lines:
                c.drawString(40, y, line)
                y -= 14
                if y < 40:
                    c.showPage()
                    y = height - 40
            c.save()

        except Exception as e:
            st.error(f"❌ Error during extraction: {e}")

# This will render after the initial analysis is done
if "raw_output" in st.session_state:
    st.subheader("📄 Medical Report")
    with st.expander("🔍 Show Details"):
        st.text_area("Summary Report", st.session_state.raw_output, height=400)

    if st.button("📥 Save Report as PDF"):
        pdf_path = Path(tempfile.gettempdir()) / "medical_summary.pdf"
        c = canvas.Canvas(str(pdf_path), pagesize=letter)
        width, height = letter

        lines = st.session_state.raw_output.splitlines()
        max_chars_per_line = 100
        formatted_lines = []
        for line in lines:
            if len(line) <= max_chars_per_line:
                formatted_lines.append(line)
            else:
                while len(line) > max_chars_per_line:
                    formatted_lines.append(line[:max_chars_per_line])
                    line = line[max_chars_per_line:]
                if line:
                    formatted_lines.append(line)

        y = height - 40
        for line in formatted_lines:
            c.drawString(40, y, line)
            y -= 14
            if y < 40:
                c.showPage()
                y = height - 40
        c.save()

        with open(pdf_path, "rb") as f:
            st.download_button("Download PDF", f, file_name="medical_summary.pdf")
            pdf_url = pdf_path.as_uri()
            st.components.v1.iframe(pdf_url, width=700, height=500)
