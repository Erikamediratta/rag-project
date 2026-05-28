import streamlit as st

from pdf_loader import PDFLoader
from text_splitter import TextSplitter
from embedding_manager import EmbeddingManager
from vector_store import VectorStore
from rag_generator import RAGGenerator



st.title("PDF RAG Chatbot")


api_key=st.secrets["GOOGLE_API_KEY"]
rag=RAGGenerator(api_key)


query = st.text_input("Ask a question from the PDF")



if st.button("Run RAG Pipeline"):

    with st.spinner("Processing PDF..."):

        # Step 1: Load PDFs
        loader = PDFLoader("text_files")
        documents = loader.load_pdfs()

        # Step 2: Split text
        splitter = TextSplitter()
        chunks = splitter.split_documents(documents)

        # Step 3: Generate embeddings
        embedding_manager = EmbeddingManager()
        embeddings = embedding_manager.generate_embeddings(chunks)

        # Step 4: Store vectors
        vector_store = VectorStore()
        vector_store.add_documents(chunks)

        # Step 5: Retrieve relevant docs
        retrieved_docs = vector_store.retrieve(query)

        # Step 6: Generate answer
        rag = RAGGenerator(api_key)
        answer = rag.generate_answer(query, retrieved_docs)

        # Display answer
        st.subheader("Answer")
        st.write(answer)

        # Show retrieved chunks
        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(retrieved_docs):

            st.write(f"### Chunk {i+1}")
            st.write(doc["content"][:1000])