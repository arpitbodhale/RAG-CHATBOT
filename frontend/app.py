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

