# audio_transcription.py
import requests
import tempfile
import os
import time
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import streamlit as st

def transcribe_audio(file_path):
    """
    Transcribe an audio file using AssemblyAI API.
    Returns (transcript, error_message).
    """
    # Load API key from environment variable
    api_key = os.getenv("ASSEMBLY_API_KEY")
    if not api_key:
        return None, "AssemblyAI API key not found in environment variables"

    base_url = "https://api.assemblyai.com"
    headers = {"authorization": api_key}

    # Upload the file
    with open(file_path, "rb") as f:
        response = requests.post(
            base_url + "/v2/upload",
            headers=headers,
            data=f
        )

    if response.status_code != 200:
        return None, f"Error uploading file: {response.text}"

    upload_url = response.json()["upload_url"]
    st.write("Debug: Audio file uploaded to AssemblyAI")

    # Start transcription process
    data = {
        "audio_url": upload_url,
        "speech_model": "universal"
    }

    url = base_url + "/v2/transcript"
    response = requests.post(url, json=data, headers=headers)

    if response.status_code != 200:
        return None, f"Error starting transcription: {response.text}"

    transcript_id = response.json()['id']
    polling_endpoint = base_url + "/v2/transcript/" + transcript_id
    st.write("Debug: Transcription started, polling for completion")

    # Poll for transcription completion
    while True:
        transcription_result = requests.get(polling_endpoint, headers=headers).json()

        if transcription_result['status'] == 'completed':
            st.write("Debug: Transcription completed")
            return transcription_result['text'], None

        elif transcription_result['status'] == 'error':
            st.write("Debug: Transcription failed")
            return None, f"Transcription failed: {transcription_result['error']}"

        time.sleep(3)

def text_to_pdf(text, output_path):
    """
    Convert text to a PDF file.
    """
    c = canvas.Canvas(output_path, pagesize=letter)
    width, height = letter
    y_position = height - 50
    line_height = 14

    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y_position, "Transcribed Clinical Conversation")
    y_position -= 30

    c.setFont("Helvetica", 10)
    # Split text into lines to fit within page width
    lines = text.split('\n')
    for line in lines:
        # Simple word wrapping
        words = line.split()
        current_line = ""
        for word in words:
            if c.stringWidth(current_line + word, "Helvetica", 10) < (width - 100):
                current_line += word + " "
            else:
                c.drawString(50, y_position, current_line.strip())
                y_position -= line_height
                current_line = word + " "
                if y_position < 50:  # Start new page if needed
                    c.showPage()
                    y_position = height - 50
                    c.setFont("Helvetica", 10)
        if current_line:
            c.drawString(50, y_position, current_line.strip())
            y_position -= line_height
        if y_position < 50:
            c.showPage()
            y_position = height - 50
            c.setFont("Helvetica", 10)

    c.showPage()
    c.save()
    st.write("Debug: Transcript converted to PDF")

def process_audio_to_pdf(uploaded_file):
    """
    Process an uploaded audio file, transcribe it, and convert to PDF.
    Returns the path to the generated PDF or (None, error_message).
    """
    # Save audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_audio:
        tmp_audio.write(uploaded_file.getvalue())
        audio_path = tmp_audio.name
    st.write("Debug: Audio file saved temporarily")

    try:
        # Transcribe audio
        transcript, error = transcribe_audio(audio_path)
        if error:
            st.write("Debug: Transcription error")
            return None, error
        st.write("Debug: Audio transcribed successfully")

        # Convert transcript to PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_pdf:
            text_to_pdf(transcript, tmp_pdf.name)
            st.write("Debug: PDF path returned")
            return tmp_pdf.name, None

    finally:
        # Clean up audio file
        try:
            os.unlink(audio_path)
            st.write("Debug: Temporary audio file cleaned up")
        except Exception:
            st.write("Debug: Failed to clean up temporary audio file")