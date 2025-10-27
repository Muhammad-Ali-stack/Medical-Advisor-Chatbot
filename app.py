import streamlit as st
from tempfile import NamedTemporaryFile
from typing import List
import os
import re

# LangChain modular imports
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.documents import Document

# --- Streamlit UI ---
st.set_page_config(page_title="Medical Disease Advisor", layout="centered")
st.title("Medical Disease Advisor")
st.write("Describe your symptoms or upload a medical report PDF to get a personalized disease analysis and recommendations.")

# --- helper to read fallback medical data file ---
MEDICAL_DATA_PATH = os.path.join("data", "medical_data.txt")
def load_medical_file_excerpt(max_chars: int = 2000) -> str:
    if not os.path.exists(MEDICAL_DATA_PATH):
        return ""
    with open(MEDICAL_DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return text[:max_chars]

# --- Identity Checker ---
def is_identity_question(q: str) -> bool:
    keywords = [
        "who made you", "who created you", "who built you", "who developed you",
        "who is your creator", "your developer", "who is your owner",
    ]
    return any(kw in q.lower() for kw in keywords)

# --- Guardrail: Detect if query is medical ---
def is_medical_query(q: str) -> bool:
    medical_keywords = [
        "symptom", "disease", "fever", "pain", "infection", "flu", "virus", "bacteria",
        "medicine", "treatment", "diagnosis", "doctor", "cough", "headache", "nausea",
        "breathing", "chest", "injury", "blood", "allergy", "diabetes", "cancer",
        "asthma", "migraine", "health", "skin", "rash", "hospital", "bp", "heart",
        "eye", "ear", "stomach", "throat", "vomit", "swelling", "painful"
    ]
    q_lower = q.lower()
    return any(re.search(rf"\b{kw}\b", q_lower) for kw in medical_keywords)

# --- Load Medical Vectorstore ---
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if not os.path.exists("vectorstore"):
        raise FileNotFoundError("Missing 'vectorstore' directory with FAISS index.")

    medical_vectorstore = FAISS.load_local(
        "vectorstore",
        embeddings,
        allow_dangerous_deserialization=True
    )
    medical_retriever = medical_vectorstore.as_retriever(search_kwargs={"k": 3})
except Exception as e:
    st.error(f"Error loading medical database. Ensure 'vectorstore' exists and has data.\n\nDetails: {e}")
    st.stop()

# --- Load LLM from Groq (fallback to Ollama) ---
try:
    groq_api_key = st.secrets.get("groq_api_key")
    if not groq_api_key:
        raise ValueError("Groq API key not found in Streamlit secrets.")

    llm = ChatOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=groq_api_key,
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=2048
    )
except Exception as e:
    st.warning(f"Groq API unavailable ({e}). Falling back to local Ollama model...")
    try:
        from langchain_community.chat_models import ChatOllama
        llm = ChatOllama(model="llama3.1:8b")
    except Exception:
        st.error("Failed to connect to any AI service. Check Groq API key or local Ollama setup.")
        st.stop()

# --- PDF Upload & Embedding ---
pdf_retriever = None
is_pdf_uploaded = False

uploaded_file = st.file_uploader("Upload a medical report (PDF)", type=["pdf"])
if uploaded_file:
    try:
        is_pdf_uploaded = True
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = splitter.split_documents(documents)

        pdf_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        pdf_vectorstore = FAISS.from_documents(docs, pdf_embeddings)
        pdf_retriever = pdf_vectorstore.as_retriever(search_kwargs={"k": 3})

        os.unlink(tmp_path)
        st.success(f"Successfully processed and embedded PDF: {uploaded_file.name}")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        is_pdf_uploaded = False

# --- Combined Document Retrieval ---
def get_combined_docs(query: str) -> List[Document]:
    docs = []
    try:
        base_docs = medical_retriever.invoke(query)
        if isinstance(base_docs, list):
            docs.extend(base_docs)
        elif base_docs:
            docs.append(base_docs)

        if pdf_retriever:
            pdf_docs = pdf_retriever.invoke(query)
            if isinstance(pdf_docs, list):
                docs.extend(pdf_docs)
            elif pdf_docs:
                docs.append(pdf_docs)
    except Exception as e:
        st.warning(f"Error during document retrieval: {e}")
        return []

    # deduplicate
    seen = set()
    unique_docs = []
    for doc in docs:
        content = getattr(doc, "page_content", None) or getattr(doc, "content", None) or ""
        if content and content not in seen:
            seen.add(content)
            unique_docs.append(doc)
    return unique_docs

# --- Medical Diagnosis Function ---
def get_medical_diagnosis(query: str, pdf_used: bool) -> str:
    try:
        relevant_docs = get_combined_docs(query)

        if not relevant_docs:
            fallback_text = load_medical_file_excerpt(max_chars=3000)
            context = fallback_text if fallback_text else ""
        else:
            context = "\n\n".join([doc.page_content for doc in relevant_docs if getattr(doc, "page_content", None)])

        if not context.strip():
            return (
                "I could not find usable medical text in the vectorstore or fallback file. "
                "Please check that 'data/medical_data.txt' is populated and that you rebuilt the vectorstore with 'python ingest.py'."
            )

        system_role = (
            "You are an experienced medical diagnosis assistant. ONLY use the CONTEXT below (do not invent facts). "
            "Provide: 1) Possible Diseases, 2) Reasoning (why), and 3) Recommended Next Steps. "
            "Always include a safety disclaimer to consult a clinician."
        )

        prompt_text = (
            f"{system_role}\n\n"
            "CONTEXT:\n{context}\n\n"
            "PATIENT SYMPTOMS / QUESTION:\n{question}\n\n"
            "Answer in numbered sections:\n1) Possible Diseases\n2) Reasoning\n3) Recommended Next Steps\n\n"
            "Keep answers concise and base them on the CONTEXT above."
        )

        prompt = PromptTemplate.from_template(prompt_text)
        chain = (
            RunnableMap({"context": RunnablePassthrough(), "question": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )

        response = chain.invoke({"context": context, "question": query})
        return response or "No response from the model."
    except Exception as e:
        st.error(f"Internal Chain Error: {str(e)}")
        return "I encountered an internal error while generating the advice."

# --- Query Input Section ---
query = st.text_input("Describe your symptoms (e.g., 'I have a high fever and shortness of breath')")

if query:
    if is_identity_question(query):
        st.success("I was developed by Muhammad Ali.")
    elif not is_medical_query(query):
        st.warning("⚠️ This assistant only answers **medical-related** questions. Please ask about symptoms, diseases, or treatments.")
    else:
        with st.spinner("Analyzing your symptoms and consulting the medical knowledge base..."):
            try:
                response = get_medical_diagnosis(query, is_pdf_uploaded)
                st.markdown("### Diagnosis (Based on Database):")
                st.write(response)
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")

# --- Footer ---
st.markdown(
    """
    <style>
    .footer {
        margin-top: 30px;
        padding: 8px;
        background-color: #f1f1f1;
        border-radius: 5px;
        text-align: center;
        color: #333;
        font-size: 13px;
    }
    a { color: #0072b1; text-decoration: none; }
    </style>
    <div class="footer">
        Made by Muhammad Ali & Hassan Siddique |
    </div>
    """,
    unsafe_allow_html=True
)
