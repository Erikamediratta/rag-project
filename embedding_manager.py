from sentence_transformers import SentenceTransformer


class EmbeddingManager:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        print("Embedding model loaded")


    def generate_embeddings(self, documents):

        texts = [
            doc.page_content
            for doc in documents
        ]

        embeddings = self.model.encode(texts)

        return embeddings