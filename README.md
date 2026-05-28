# RAG PDF Chatbot

A simple RAG (Retrieval-Augmented Generation) chatbot that answers questions from PDF documents using Gemini API.

## Features

* Load PDF files
* Split text into chunks
* Create embeddings
* Store embeddings in ChromaDB
* Ask questions from PDFs
* Streamlit user interface

## Tech Stack

* Python
* LangChain
* ChromaDB
* Sentence Transformers
* Gemini API
* Streamlit

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

## Project Flow

PDF → Text Splitting → Embeddings → Vector Database → Retrieval → Gemini Response

## Author

Erika Mediratta
