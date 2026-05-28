import streamlit as st

from pdf_loader import PDFLoader
from text_splitter import TextSplitter
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from rag_generator import RAGGenerator


st.title("PDF RAG Chatbot")

api_key = st.secrets["GOOGLE_API_KEY"]


@st.cache_resource
def load_pipeline():
    loader = PDFLoader("text_files")
    documents = loader.load_pdfs()

    splitter = TextSplitter()
    chunks = splitter.split_documents(documents)

    embedding_manager = EmbeddingManager()
    embedding_manager.generate_embeddings(chunks)

    vector_store = VectorStore()
    vector_store.add_documents(chunks)

    return vector_store



@st.cache_resource
def load_rag():
    return RAGGenerator(api_key)


vector_store = load_pipeline()
rag = load_rag()


query = st.text_input("Ask a question from the PDF")

if st.button("Get Answer"):

    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Thinking..."):

            try:
                retrieved_docs = vector_store.retrieve(query)
                answer = rag.generate_answer(query, retrieved_docs)

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Retrieved Chunks")
                for i, doc in enumerate(retrieved_docs):
                    st.write(f"### Chunk {i+1}")
                    st.write(doc["content"][:1000])

            except Exception as e:
                st.error(f"Error: {str(e)}")