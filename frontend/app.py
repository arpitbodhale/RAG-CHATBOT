# import streamlit as st
# import requests
# import os

# # FastAPI backend URL
# BACKEND_URL = "http://127.0.0.1:8000"

# # Set page config
# st.set_page_config(page_title="Constitution of India Q&A", layout="wide")

# # Background image styling
# page_bg = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Constitution_of_India.jpg/800px-Constitution_of_India.jpg");
#     background-size: cover;
#     background-position: center;
# }
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # Title and description
# st.title("📜 Constitution of India - Q&A")
# st.markdown("Upload a PDF and ask questions to extract relevant information.")

# # File uploader
# uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
# if uploaded_file is not None:
#     file_path = os.path.join("temp", uploaded_file.name)
#     os.makedirs("temp", exist_ok=True)
    
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
    
#     # Upload PDF to backend
#     with st.spinner("Uploading PDF and processing..."):
#         files = {"file": open(file_path, "rb")}
#         response = requests.post(f"{BACKEND_URL}/upload/", files=files)
#         if response.status_code == 200:
#             st.success("PDF uploaded successfully!")
#         else:
#             st.error("Failed to upload PDF.")

# # Query input
# query = st.text_input("Ask a question about the document:")
# if st.button("Get Answer") and query and uploaded_file:
#     pdf_name = uploaded_file.name
#     payload = {"query": query, "pdf_name": pdf_name}
    
#     with st.spinner("Fetching answer..."):
#         response = requests.post(f"{BACKEND_URL}/query/", json=payload)
#         if response.status_code == 200:
#             answer = response.json().get("response", "No answer found.")
#             st.markdown(f"### Answer:")
#             st.write(answer)
#         else:
#             st.error("Failed to fetch answer.")


# -- working code 
# import streamlit as st
# import requests
# import os

# # FastAPI backend URL
# BACKEND_URL = "http://127.0.0.1:8000"

# # Set page config
# st.set_page_config(page_title="Constitution of India Q&A", layout="wide")

# # Background image styling
# page_bg = """
# <style>
# [data-testid="stAppViewContainer"] {
#     background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Constitution_of_India.jpg/800px-Constitution_of_India.jpg");
#     background-size: cover;
#     background-position: center;
# }
# </style>
# """
# st.markdown(page_bg, unsafe_allow_html=True)

# # Title and description
# st.title("📜 Constitution of India - Q&A")
# st.markdown("Upload a PDF or ask questions from already stored data.")

# # Check if vector index exists
# response = requests.get(f"{BACKEND_URL}/available_documents/")
# if response.status_code == 200:
#     available_docs = response.json().get("documents", [])
# else:
#     available_docs = []

# # Select from previously uploaded PDFs
# selected_doc = None
# if available_docs:
#     selected_doc = st.selectbox("Choose a document to query:", available_docs)

# # File uploader (optional)
# uploaded_file = st.file_uploader("Upload a new PDF document (optional)", type=["pdf"])
# if uploaded_file is not None:
#     file_path = os.path.join("temp", uploaded_file.name)
#     os.makedirs("temp", exist_ok=True)

#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Upload PDF to backend
#     with st.spinner("Uploading PDF and processing..."):
#         files = {"file": open(file_path, "rb")}
#         response = requests.post(f"{BACKEND_URL}/upload/", files=files)
#         if response.status_code == 200:
#             st.success("PDF uploaded successfully!")
#             selected_doc = uploaded_file.name  # Use the newly uploaded file
#         else:
#             st.error("Failed to upload PDF.")

# # Query input
# query = st.text_input("Ask a question about the document:")
# if st.button("Get Answer") and query:
#     if not selected_doc:
#         st.error("Please select an existing document or upload a new one.")
#     else:
#         payload = {"query": query, "pdf_name": selected_doc}

#         with st.spinner("Fetching answer..."):
#             response = requests.post(f"{BACKEND_URL}/query/", json=payload)
#             if response.status_code == 200:
#                 answer = response.json().get("response", "No answer found.")
#                 st.markdown(f"### Answer:")
#                 st.write(answer)
#             else:
#                 st.error("Failed to fetch answer.")
# #############################################################################################################################
# -- this query section is another enchaced section
# # Query input
# query = st.text_input("Ask a question about the document:")
# if st.button("Get Answer") and query:
#     payload = {"query": query}

#     with st.spinner("Fetching answer..."):
#         response = requests.post(f"{BACKEND_URL}/query/", json=payload)
#         if response.status_code == 200:
#             answer = response.json().get("response", "No answer found.")
#             st.markdown(f"### Answer:")
#             st.write(answer)
#         else:
#             st.error("Failed to fetch answer.")

import streamlit as st
import requests

# FastAPI backend URL
BACKEND_URL = "http://127.0.0.1:8000"

# Set page config
st.set_page_config(page_title="Constitution of India Q&A", layout="wide")

# Background image styling
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Constitution_of_India.jpg/800px-Constitution_of_India.jpg");
    background-size: cover;
    background-position: center;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# Title and description
st.title("📜 Q&A CHATBOT")
st.markdown("Ask any question about the Constitution. The system will automatically retrieve relevant information.")

# Query input
query = st.text_input("Ask a question:")
if st.button("Get Answer") and query:
    payload = {"query": query}  # No need to send `pdf_name`

    with st.spinner("Fetching answer..."):
        response = requests.post(f"{BACKEND_URL}/query/", json=payload)
    
    if response.status_code == 200:
        answer = response.json().get("response", "No relevant answer found. Please add a relevant document.")
        st.markdown(f"### Answer:")
        st.write(answer)
    else:
        st.error("Failed to fetch answer.")

# File uploader (for adding new PDFs)
uploaded_file = st.file_uploader("Upload a new PDF document (optional)", type=["pdf"])
if uploaded_file is not None:
    files = {"file": uploaded_file}
    
    with st.spinner("Uploading and processing the document..."):
        response = requests.post(f"{BACKEND_URL}/upload/", files=files)
    
    if response.status_code == 200:
        st.success("PDF uploaded and processed successfully!")
    else:
        st.error("Failed to upload PDF.")

