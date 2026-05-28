import chromadb
from sentence_transformers import SentenceTransformer


class VectorStore:

    def __init__(self):

       
        self.client = chromadb.PersistentClient(path="/tmp/chroma_db")

        self.collection = self.client.get_or_create_collection(
            name="pdf_documents"
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

    def add_documents(self, documents):

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        ids = [str(i) for i in range(len(texts))]

        # ✅ generate embeddings here (NOT outside)
        embeddings = self.embedding_model.encode(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        print(f"{len(texts)} documents added")

    def retrieve(self, query, top_k=3):

        query_embedding = self.embedding_model.encode(query).tolist()

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        retrieved_docs = []

        for i in range(len(results["documents"][0])):

            retrieved_docs.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i]
            })

        return retrieved_docs