import os
import time
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
os.environ["GGROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

# Initialize API key for Groq
groq_api_key = os.getenv("GROQ_API_KEY")

# Groq LLM initialization
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="Llama3-8b-8192"
)

# Set up prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the context without mentioning according to context in answer. 
    Please provide the most accurate and elaborated answer based on the question.
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Streamlit UI setup
st.title("Conversational PDF Q&A Bot")
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Function to create vector embeddings from uploaded PDFs
def create_vector_embeddings():
    # Ensure session state is initialized for documents and vectors
    if "documents" not in st.session_state:
        st.session_state.documents = []  # Initialize documents as an empty list
    if "vectors" not in st.session_state:
        st.session_state.vectors = None  # Initialize vectors as None

    # Initialize embeddings
    st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # File uploader for PDFs
    if uploaded_files:
        # Clear the documents before adding new ones
        st.session_state.documents.clear()

        # Load documents from uploaded PDFs
        for uploaded_file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            st.session_state.loader = PyPDFLoader(temppdf)
            st.session_state.docs = st.session_state.loader.load()
            st.session_state.documents.extend(st.session_state.docs)  # Add to session state

        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.documents)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
        st.write("Vector embeddings created successfully! You are ready to ask your questions now.")
    else:
        st.warning("Please upload PDF files to create vector embeddings.")

# Call function to create vector embeddings when "Submit PDFs" is clicked
if st.button("Submit PDFs"):
    create_vector_embeddings()

# Conditional user input for querying the PDFs
if "vectors" in st.session_state and st.session_state.vectors is not None:
    # Allow user to input questions only if the vectors are ready
    user_prompt = st.text_input("Enter your question about PDFs:")

    # If the user provides a query
    if user_prompt:
        # Create retrieval chain
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Process and time the response
        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_prompt})
        end = time.process_time()

        # Display response time and answer
        st.write(response["answer"])
        st.write(f"Response time: {end - start} seconds")


else:
    # Inform the user to upload files before entering queries
    st.info("INFO: Please upload PDFs and click 'Submit PDFs' to before entering your question.")
