# # -- working code 
# import os
# import hashlib
# import logging
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Initialize logging
# logger = logging.getLogger(__name__)

# # Embedding model
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# VECTOR_DIR = "data/vector_indices"
# os.makedirs(VECTOR_DIR, exist_ok=True)

# def get_vector_store(pdf_path):
#     """Retrieve or create FAISS vector store for the PDF."""
#     try:
#         file_stats = os.stat(pdf_path)
#         pdf_id = f"{os.path.basename(pdf_path)}_{file_stats.st_size}_{file_stats.st_mtime}"
#         pdf_hash = hashlib.md5(pdf_id.encode()).hexdigest()
#         index_path = os.path.join(VECTOR_DIR, pdf_hash)

#         if os.path.exists(index_path):
#             logger.info(f"Loading existing vector store from {index_path}")
#             return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

#         logger.info(f"Creating new vector store for {pdf_path}")
#         loader = PyPDFLoader(pdf_path)
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
#         texts = text_splitter.split_documents(documents)
#         vector_store = FAISS.from_documents(texts, embedding_model)
#         vector_store.save_local(index_path)
#         return vector_store

#     except Exception as e:
#         logger.error(f"Error in get_vector_store: {e}")
#         raise
 ############################################################################################################################

# def get_vector_store(pdf_path=None, pdf_name=None):
#     """Retrieve or create FAISS vector store for the PDF."""
#     try:
#         if pdf_name:  # Allow fetching vector store just by name
#             pdf_hash = hashlib.md5(pdf_name.encode()).hexdigest()
#             index_path = os.path.join(VECTOR_DIR, pdf_hash)
#         else:
#             file_stats = os.stat(pdf_path)
#             pdf_id = f"{os.path.basename(pdf_path)}_{file_stats.st_size}_{file_stats.st_mtime}"
#             pdf_hash = hashlib.md5(pdf_id.encode()).hexdigest()
#             index_path = os.path.join(VECTOR_DIR, pdf_hash)

#         if os.path.exists(index_path):
#             logger.info(f"Loading existing vector store from {index_path}")
#             return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

#         if pdf_path:  # Process if no existing index
#             logger.info(f"Creating new vector store for {pdf_path}")
#             loader = PyPDFLoader(pdf_path)
#             documents = loader.load()
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
#             texts = text_splitter.split_documents(documents)
#             vector_store = FAISS.from_documents(texts, embedding_model)
#             vector_store.save_local(index_path)
#             return vector_store
#         else:
#             raise FileNotFoundError("Vector store not found and no PDF provided for processing.")

#     except Exception as e:
#         logger.error(f"Error in get_vector_store: {e}")
#         raise


import os
import hashlib
import logging
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize logging
logger = logging.getLogger(__name__)

# Embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

VECTOR_DIR = "data/vector_indices"
os.makedirs(VECTOR_DIR, exist_ok=True)

def get_vector_store(pdf_path):
    """Create or retrieve FAISS vector store for a specific PDF."""
    try:
        file_stats = os.stat(pdf_path)
        pdf_id = f"{os.path.basename(pdf_path)}_{file_stats.st_size}_{file_stats.st_mtime}"
        pdf_hash = hashlib.md5(pdf_id.encode()).hexdigest()
        index_path = os.path.join(VECTOR_DIR, pdf_hash)

        if os.path.exists(index_path):
            logger.info(f"Loading existing vector store from {index_path}")
            return FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)

        logger.info(f"Creating new vector store for {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embedding_model)
        vector_store.save_local(index_path)
        return vector_store

    except Exception as e:
        logger.error(f"Error in get_vector_store: {e}")
        raise

def search_across_vectors(query, k=3):
    """Search for relevant embeddings across all stored vector indices."""
    results = []
    for index_file in os.listdir(VECTOR_DIR):
        index_path = os.path.join(VECTOR_DIR, index_file)
        if os.path.isdir(index_path):
            try:
                vector_store = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
                results.extend(vector_store.similarity_search(query, k=k))
            except Exception as e:
                logger.error(f"Error searching in {index_path}: {e}")
    return results
