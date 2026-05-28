from langchain_community.document_loaders import PyPDFLoader
import os


class PDFLoader:

    def __init__(self, folder_path):

        self.folder_path = folder_path


    def load_pdfs(self):

        documents = []

        pdf_files = [
            file for file in os.listdir(self.folder_path)
            if file.endswith(".pdf")
        ]

        print(f"Found {len(pdf_files)} PDF files to process")

        for file in pdf_files:

            pdf_path = os.path.join(self.folder_path, file)

            loader = PyPDFLoader(pdf_path)

            docs = loader.load()

            documents.extend(docs)

            print(f"Loaded {len(docs)} pages from {file}")

        return documents