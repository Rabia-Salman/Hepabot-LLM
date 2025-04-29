from typing import Dict, Any, Optional
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma


def create_rag_chain(vector_db: Chroma,
                     model_name: str = "llama3.2",
                     top_k: int = 4) -> Any:
    """
    Create a RAG (Retrieval Augmented Generation) chain for medical diagnosis.

    Args:
        vector_db: Vector database to retrieve from
        model_name: Name of the LLM model to use
        top_k: Number of documents to retrieve

    Returns:
        A runnable chain that can answer medical diagnosis questions
    """
    # Initialize LLM
    llm = ChatOllama(model=model_name)

    # Create multi-query retriever for better results
    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant specialized in medical terminology. 
        Your task is to generate five different versions of the given user question to retrieve 
        relevant medical documents from a vector database. By generating multiple perspectives 
        on the user question, your goal is to help retrieve the most relevant medical information.
        Provide these alternative questions separated by newlines.

        Original question: {question}""",
    )

    # Create retriever with multi-query capability for better recall
    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(search_kwargs={"k": top_k}),
        llm,
        prompt=query_prompt
    )

    # Define RAG prompt template
    template = """Answer the question based ONLY on the following context:
    {context}

    Based on the question, retrieve a diagnosis from the context being very precise and brief.
    Write symptoms, lab tests and medications that are associated with the diagnosis in the documents.

    Follow this format:
    Diagnosis: [provide the most likely diagnosis based on the context]
    Typical Symptoms: [list the main symptoms associated with this diagnosis]
    Lab Tests Recommended: [list appropriate lab tests to confirm diagnosis]
    Medications Recommended: [list medications mentioned for this condition]

    If you are not confident in the diagnosis, clearly state this.
    Do not add introductory or conclusive remarks.

    Question: {question}
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain
    chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    return chain


def ask_question(chain: Any, question: str) -> str:
    """
    Ask a question to the RAG chain.

    Args:
        chain: The RAG chain to use
        question: The question to ask

    Returns:
        The response from the chain
    """
    try:
        response = chain.invoke(question)
        return response
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        print(error_message)
        return error_message