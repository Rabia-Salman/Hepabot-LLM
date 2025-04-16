# extract_fields.py
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from utils import load_conversations_from_pdf

llm = ChatOllama(model="llama3.2")

extraction_template = ChatPromptTemplate.from_template("""
You are an expert medical assistant who prepares structured notes for a doctor.

From the clinical conversation below, extract and organize the information under clearly labeled headings and subheadings.

Use the following format (no markdown, no bullet points, no symbols like '-', '*', or '#'):

Symptoms Present:
<list all symptoms present, comma separated or in plain text>

Symptoms Absent:
<list all symptoms absent, including those explicitly denied>

Diagnosis:
<diagnosis>

Medications Prescribed:
<list medications>

Surgical Plans:
<describe any surgical plans>

Past Medical History:
<relevant history>

Lab and Imaging Findings:
<list findings>

Follow-up:
<follow up visit and instruction>
Patient Visits Summary:
<short narrative summary>

Conversation:
{conversation}

Strict Instructions:
- DO NOT SKIP ANY SYMPTOMS OR LAB REPORTS
- INCLUDE SYMPTOMS OR CONDITIONS EVEN IF STATED AS ABSENT OR DENIED
- NO BULLET POINTS, NO MARKDOWN, NO SYMBOLS — only plain text under clear headings
- Do not write full conversation into PDF
- KEEP IT CONCISE AND STRUCTURED
""")

# Load conversation
turns = load_conversations_from_pdf("../Clinical-rag/data/history_physical_1.pdf")
conversation_text = "\n".join([f"{t['speaker']}: {t['text']}" for t in turns])

chain = extraction_template | llm | StrOutputParser()
output = chain.invoke({"conversation": conversation_text})
print(output)
