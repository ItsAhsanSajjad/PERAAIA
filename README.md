**PERA AI Assistant**

The PERA AI Assistant is a document-grounded, retrieval-augmented AI system designed to answer questions strictly from official PERA policies, Acts, and notifications. It ensures accurate, verifiable, and non-hallucinatory responses with clear document references.

**Key Features**

Document-Based Answers Only
Responds strictly using indexed PERA documents (PDF/DOCX), with page-level citations.

Automatic Document Ingestion
New or updated documents placed in the data directory are automatically detected and indexed.

Greeting & Small-Talk Handling
Deterministic intent handling for greetings (English, Urdu, Roman Urdu) without triggering document retrieval.

Voice & Text Queries
Supports both typed and voice-based questions via a Streamlit interface.

Hallucination Prevention
Built-in refusal logic when relevant evidence is not found.

**High-Level Architecture**

User submits a query (text or voice)

Greeting/small-talk intent is detected first

Relevant document chunks are retrieved using FAISS

AI generates an answer strictly from retrieved evidence

Verified answer is returned with references

**Technology Stack**

Frontend: Streamlit

Backend: Python

Search & Retrieval: FAISS

AI Models: OpenAI (RAG-based)

Document Parsing: PDF & DOCX extractors

**Usage**

Place PERA documents in assets/data/

Run the application:

streamlit run app.py


Ask questions related to PERA policies and regulations.


**Scope & Disclaimer**


This assistant is designed only for PERA-related queries and does not provide legal advice or information beyond the provided documents.


# ask.pera
