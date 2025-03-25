import logging
from openai import OpenAI
# from vector_store import get_vector_store
from backend.vector_store import get_vector_store
from backend.config import NVIDIA_API_KEY

# Initialize logging
logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)

def process_pdf(query, pdf_path):
    """Processes the query and fetches relevant answers from vector store."""
    try:
        vector_store = get_vector_store(pdf_path)
        results = vector_store.similarity_search(query, k=3)
        if not results:
            return {"response": "No relevant information found."}

        context = "\n".join([f"(Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for doc in results])
        prompt = f"""
        You are an AI assistant with document retrieval. Use the following retrieved context to answer the user's question:
        
        Context:
        {context}
        
        Question: {query}
        Answer:
        """

        completion = client.chat.completions.create(
            model="meta/llama3-70b-instruct",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=512,
            stream=True
        )

        response = ""
        for chunk in completion:
            if chunk.choices[0].delta and chunk.choices[0].delta.content:
                response += chunk.choices[0].delta.content

        return {"response": response}

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return {"error": "Error generating response"}
