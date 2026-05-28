import google.generativeai as genai


class RAGGenerator:

    def __init__(self, api_key):

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name="models/gemini-1.5-flash"
        )


    def generate_answer(
        self,
        query,
        retrieved_docs
    ):

        context = ""

        for doc in retrieved_docs:

            context += doc["content"] + "\n\n"

        prompt = f"""
        Answer ONLY from the context below.

        Context:
        {context}

        Question:
        {query}
        """

        response = self.model.generate_content(
            prompt
        )

        return response.text