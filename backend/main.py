#-- this code is just to connect with the fastapi 
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import os
# import shutil
# import logging
# from pydantic import BaseModel
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from openai import OpenAI
# import hashlib
# from config import NVIDIA_API_KEY  # Ensure your key is correctly set in config.py or env vars

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Initialize FastAPI app
# app = FastAPI()

# # Initialize OpenAI client (NVIDIA API)
# try:
#     client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=NVIDIA_API_KEY)
#     logger.info("Successfully connected to NVIDIA API.")
# except Exception as e:
#     logger.error(f"Failed to initialize NVIDIA API: {e}")

# # Initialize embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# UPLOAD_DIR = "uploaded_pdfs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# def get_vector_store(pdf_path):
#     """Get or create vector store based on PDF hash."""
#     try:
#         file_stats = os.stat(pdf_path)
#         pdf_id = f"{os.path.basename(pdf_path)}_{file_stats.st_size}_{file_stats.st_mtime}"
#         pdf_hash = hashlib.md5(pdf_id.encode()).hexdigest()
#         index_dir = "vector_indices"
#         os.makedirs(index_dir, exist_ok=True)
#         index_path = os.path.join(index_dir, pdf_hash)
        
#         # Update mapping file
#         mapping_file = os.path.join(index_dir, "pdf_mapping.txt")
#         with open(mapping_file, "a+") as f:
#             f.seek(0)
#             if not any(pdf_hash in line for line in f):
#                 f.write(f"{pdf_hash}: {pdf_path}\n")
        
#         if os.path.exists(index_path):
#             logger.info(f"Loading existing vector store from {index_path}")
#             return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
#         else:
#             logger.info(f"Creating new vector store for {pdf_path}")
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
#             texts = text_splitter.split_documents(documents)
#             vector_store = FAISS.from_documents(texts, embedding_model)
#             vector_store.save_local(index_path)
#             return vector_store
#     except Exception as e:
#         logger.error(f"Error in get_vector_store: {e}")
#         raise

# class QueryRequest(BaseModel):
#     query: str
#     pdf_name: str

# @app.post("/upload/")
# async def upload_pdf(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         vector_store = get_vector_store(file_path)
#         return JSONResponse(content={"message": "PDF uploaded and processed successfully", "pdf_name": file.filename})
#     except Exception as e:
#         logger.error(f"Error uploading PDF: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.post("/query/")
# async def query_document(request: QueryRequest):
#     pdf_path = os.path.join(UPLOAD_DIR, request.pdf_name)
#     if not os.path.exists(pdf_path):
#         return JSONResponse(content={"error": "PDF not found"}, status_code=404)
    
#     try:
#         vector_store = get_vector_store(pdf_path)
#         results = vector_store.similarity_search(request.query, k=3)
#         if not results:
#             return JSONResponse(content={"response": "No relevant information found."})
        
#         context = "\n".join([f"(Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for doc in results])
#         prompt = f"""
#         You are an AI assistant with document retrieval. Use the following retrieved context to answer the user's question:
        
#         Context:
#         {context}
        
#         Question: {request.query}
#         Answer:
#         """
        
#         completion = client.chat.completions.create(
#             model="meta/llama3-70b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=512,
#             stream=True
#         )
        
#         response = ""
#         for chunk in completion:
#             if chunk.choices[0].delta and chunk.choices[0].delta.content:
#                 response += chunk.choices[0].delta.content
        
#         return JSONResponse(content={"response": response})
    
#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         return JSONResponse(content={"error": "Error generating response"}, status_code=500)

# @app.get("/")
# def home():
#     return {"message": "Welcome to the FastAPI-powered RAG system!"}
# ##########################################################################################################################################
# -- working code 
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import os
# import shutil
# import logging
# from pydantic import BaseModel
# # from config import NVIDIA_API_KEY
# from backend.config import NVIDIA_API_KEY  # Use absolute import
# # from vector_store import get_vector_store
# from backend.vector_store import get_vector_store
# #from utils import process_pdf
# from backend.utils import process_pdf

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app
# app = FastAPI()

# # Upload directory
# UPLOAD_DIR = "data/uploaded_pdfs"
# os.makedirs(UPLOAD_DIR, exist_ok=True)

# class QueryRequest(BaseModel):
#     query: str
#     pdf_name: str

# @app.post("/upload/")
# async def upload_pdf(file: UploadFile = File(...)):
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)
        
#         get_vector_store(file_path)
#         return JSONResponse(content={"message": "PDF uploaded successfully!", "pdf_name": file.filename})
#     except Exception as e:
#         logger.error(f"Error uploading PDF: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.post("/query/")
# async def query_document(request: QueryRequest):
#     pdf_path = os.path.join(UPLOAD_DIR, request.pdf_name)
#     if not os.path.exists(pdf_path):
#         return JSONResponse(content={"error": "PDF not found"}, status_code=404)
    
#     return process_pdf(request.query, pdf_path)

# -- this query section is another enchaced section
# query = st.text_input("Ask a question about the document:")
# if st.button("Get Answer") and query:
#     if not selected_doc:
#         st.error("Please select an existing document or upload a new one.")
#     else:
#         payload = {"query": query, "pdf_name": selected_doc}  # Ensure pdf_name is included

#         with st.spinner("Fetching answer..."):
#             response = requests.post(f"{BACKEND_URL}/query/", json=payload)
#             if response.status_code == 200:
#                 answer = response.json().get("response", "No answer found.")
#                 st.markdown(f"### Answer:")
#                 st.write(answer)
#             else:
#                 st.error("Failed to fetch answer.")


# @app.get("/")
# def home():
#     return {"message": "Welcome to the FastAPI-powered RAG system!"}

# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import JSONResponse
# import os
# import shutil
# import logging
# from pydantic import BaseModel
# from backend.config import NVIDIA_API_KEY  # Use absolute import
# from backend.vector_store import get_vector_store, search_across_vectors
# from backend.utils import process_pdf
# #from backend.llm import client  # Ensure you have an LLM client setup

# # Initialize logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # FastAPI app
# app = FastAPI()

# # Directories
# UPLOAD_DIR = "data/uploaded_pdfs"
# VECTOR_DIR = "data/vector_store"
# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(VECTOR_DIR, exist_ok=True)

# class QueryRequest(BaseModel):
#     query: str

# @app.post("/upload/")
# async def upload_pdf(file: UploadFile = File(...)):
#     """Uploads a PDF, processes it, and stores embeddings in the vector database."""
#     file_path = os.path.join(UPLOAD_DIR, file.filename)
#     try:
#         with open(file_path, "wb") as buffer:
#             shutil.copyfileobj(file.file, buffer)

#         # Process PDF and store in vector DB
#         get_vector_store(file_path)

#         return JSONResponse(content={"message": "PDF uploaded successfully!", "pdf_name": file.filename})
#     except Exception as e:
#         logger.error(f"Error uploading PDF: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.get("/available_documents/")
# async def available_documents():
#     """Returns a list of available processed PDFs."""
#     try:
#         stored_files = [f for f in os.listdir(VECTOR_DIR) if os.path.isdir(os.path.join(VECTOR_DIR, f))]
#         return {"documents": stored_files}
#     except Exception as e:
#         logger.error(f"Error fetching available documents: {e}")
#         return JSONResponse(content={"error": str(e)}, status_code=500)

# @app.post("/query/")
# async def query_document(request: QueryRequest):
#     """Process query across all vector indices and return an AI-generated answer."""
#     try:
#         results = search_across_vectors(request.query, k=3)

#         if not results:
#             return JSONResponse(content={"response": "No relevant information found."})

#         # Extract relevant text from retrieved chunks
#         context = "\n".join([
#             f"(Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for doc in results
#         ])

#         # Construct LLM prompt
#         prompt = f"""
#         You are an AI assistant with document retrieval.
#         Use the following retrieved context to answer the user's question:
#         Context: {context}
#         Question: {request.query}
#         Answer:
#         """

#         # Generate response using LLaMA 3
#         completion = client.chat.completions.create(
#             model="meta/llama3-70b-instruct",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.3,
#             max_tokens=512,
#             stream=True
#         )

#         response = ""
#         for chunk in completion:
#             if chunk.choices[0].delta and chunk.choices[0].delta.content:
#                 response += chunk.choices[0].delta.content

#         return JSONResponse(content={"response": response})

#     except Exception as e:
#         logger.error(f"Error processing query: {e}")
#         return JSONResponse(content={"error": "Error generating response"}, status_code=500)

# @app.get("/")
# def home():
#     """Home route"""
#     return {"message": "Welcome to the FastAPI-powered RAG system!"}

#main.py

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import os
import shutil
import logging
import requests
from pydantic import BaseModel
from backend.config import NVIDIA_API_KEY  # Import API key
from backend.vector_store import get_vector_store, search_across_vectors
from backend.utils import process_pdf

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI()

# Directories
UPLOAD_DIR = "data/uploaded_pdfs"
# VECTOR_DIR = "data/vector_store"
os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(VECTOR_DIR, exist_ok=True)

class QueryRequest(BaseModel):
    query: str

@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    """Uploads a PDF, processes it, and stores embeddings in the vector database."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process PDF and store in vector DB
        get_vector_store(file_path)

        return JSONResponse(content={"message": "PDF uploaded successfully!", "pdf_name": file.filename})
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/available_documents/")
async def available_documents():
    """Returns a list of available processed PDFs."""
    try:
        stored_files = [f for f in os.listdir(VECTOR_DIR) if os.path.isdir(os.path.join(VECTOR_DIR, f))]
        return {"documents": stored_files}
    except Exception as e:
        logger.error(f"Error fetching available documents: {e}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/query/")
async def query_document(request: QueryRequest):
    """Process query across all vector indices and return an AI-generated answer."""
    try:
        results = search_across_vectors(request.query, k=3)

        if not results:
            return JSONResponse(content={"response": "No relevant information found."})

        # Extract relevant text from retrieved chunks
        context = "\n".join([
            f"(Page {doc.metadata.get('page', 'N/A')}): {doc.page_content}" for doc in results
        ])

        # Construct LLM prompt
        prompt = f"""
        You are an AI assistant with document retrieval.
        Use the following retrieved context to answer the user's question:
        Context: {context}
        Question: {request.query}
        Answer:
        """

        # Validate API Key
        if not NVIDIA_API_KEY:
            logger.error("NVIDIA API Key is missing or invalid!")
            return JSONResponse(content={"error": "NVIDIA API Key is missing."}, status_code=500)

        # NVIDIA LLaMA 3 API Call
        headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta/llama-3.3-70b-instruct",  # ✅ Fixed model name
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.3,
            "max_tokens": 512
        }

        response = requests.post("https://integrate.api.nvidia.com/v1/chat/completions", headers=headers, json=payload)  # ✅ Fixed API URL

        if response.status_code == 200:
            llm_response = response.json()["choices"][0]["message"]["content"]
            return JSONResponse(content={"response": llm_response})
        else:
            logger.error(f"NVIDIA API error: {response.status_code}, Response: {response.text}")
            return JSONResponse(content={"error": f"API error: {response.text}"}, status_code=500)

    except requests.exceptions.RequestException as req_err:
        logger.error(f"Request failed: {req_err}")
        return JSONResponse(content={"error": "Failed to connect to NVIDIA API"}, status_code=500)

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(content={"error": "Error generating response"}, status_code=500)

@app.get("/")
def home():
    """Home route"""
    return {"message": "Welcome to the FastAPI-powered RAG system!"}

