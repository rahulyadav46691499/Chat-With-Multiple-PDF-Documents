import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from config import GOOGLE_API_KEY  # Import API key from config.py

# Configure Google Gemini API
key = genai.configure(api_key=GOOGLE_API_KEY)
print(key)


# Function to extract text from uploaded PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle NoneType error
    return text


# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Function to create and save a FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY,model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


# Function to create a conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, reply with "Answer is not available in the context."
    
    Context: {context}
    Question: {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY,model="gemini-pro")
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to process user input and return AI-generated response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(google_api_key=GOOGLE_API_KEY,model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    
    st.write("Reply:", response["output_text"])


# Streamlit UI setup
st.set_page_config(page_title="Chat with PDF", layout="wide")
st.header("📄 Chat with PDF using Gemini AI 💁")

# User question input
user_question = st.text_input("Ask a Question from the PDF Files")
if user_question:
    user_input(user_question)

# Sidebar for PDF upload
with st.sidebar:
    st.title("📂 Upload PDFs:")
    pdf_docs = st.file_uploader("Upload your PDF files", accept_multiple_files=True)

    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.success("Done ✅ PDF processed!")
