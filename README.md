# Medical Disease Advisor (RAG Chatbot)

## About

This project is a **Medical Disease Advisor Chatbot** built using a **Retrieval-Augmented Generation (RAG)** architecture.  
It combines **Groq’s LLaMA 3.1 model** (via API) with a **custom medical dataset** stored in a **FAISS vector database** to provide medically-informed, context-aware responses.  

If the Groq API is unavailable, the chatbot automatically falls back to a **local Ollama model**.  
Built-in safety mechanisms ensure the chatbot **does not provide medical treatment advice**, focusing only on **diagnostic insights** and next-step recommendations.

---

## Features

- **Retrieval-Augmented Generation (RAG)** for grounded, accurate responses  
-  **Streamlit-based web UI** for easy use  
-  **FAISS vector database** for fast retrieval  
-  **PDF upload** support for analyzing medical reports  
-  **Automatic fallback** from Groq API to local Ollama model  
-  **Safety Guardrails** — only provides diagnostic reasoning, not treatment


### Installation  
1. Clone the repo:  
   ```bash
   git clone https://github.com/Muhammad-Ali-stack/Medical-Advisor-Chatbot.git
   cd Medical-Advisor-Chatbot
2. pip install -r requirements.txt
3. In the secrets.toml file in .streamlit folder paste groq api key.
4. python ingest.py
5. python app.py