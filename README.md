# Resume Screening Chatbot (AI/ML – RAG Based)

This project is an AI-powered Resume Screening Chatbot built using
Retrieval Augmented Generation (RAG).  
It analyzes a resume PDF and answers questions related to skills,
strengths, missing skills, and suitability for AI/ML internships.

---

## 🔹 Project Overview
The chatbot converts a resume PDF into semantic embeddings and stores
them in a FAISS vector database.  
When a user asks a question, the system retrieves the most relevant
resume sections and uses them to generate accurate responses.

---

## 🔹 Features
- Load resume in PDF format
- Split resume text into chunks
- Generate semantic embeddings
- Store embeddings using FAISS
- Answer user queries about the resume
- Suitable for AI/ML internship screening

---

## 🔹 Tech Stack
- Python
- LangChain
- FAISS
- HuggingFace Embeddings
- PyPDF

---

## 🔹 Key Concepts Used
- Retrieval Augmented Generation (RAG)
- Vector Embeddings
- Semantic Search
- Natural Language Processing (NLP)

---

## 🔹 How It Works
1. Resume PDF is loaded and processed
2. Text is split into overlapping chunks
3. Chunks are converted into embeddings
4. Embeddings are stored in FAISS
5. User asks a question
6. Relevant resume sections are retrieved and displayed

---

## 🔹 How to Run
1. Install required libraries
2. Place your resume as `resume.pdf` in the project folder
3. Run the application:

```bash
python app.py
