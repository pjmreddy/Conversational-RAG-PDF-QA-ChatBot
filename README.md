![GitHub License](https://img.shields.io/github/license/yourusername/RAG-bot-with-GROQAPI?style=flat-square)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue?style=flat-square&logo=python)
![GROQ Powered](https://img.shields.io/badge/GROQ-API-01B4A4?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij48cGF0aCBkPSJNMTIgMEM1LjM3MyAwIDAgNS4zNzMgMCAxMnM1LjM3MyAxMiAxMiAxMiAxMi01LjM3MyAxMi0xMlMxOC42MjcgMCAxMiAwem0wIDIyLjI5M2MtNS42MjYgMC0xMC4yOTMtNC42NjctMTAuMjkzLTEwLjI5M1M2LjM3NCAxLjcgMTIgMS43czEwLjI5MyA0LjY2NyAxMC4yOTMgMTAuMjkzLTQuNjY3IDEwLjI5My0xMC4yOTMgMTAuMjkzem0yLjM1LTguNTY2bC0yLjI5IDIuMjg4aC0uMDAybC0yLjI5LTIuMjg4LTQuMTQzIDQuMTQzIDIuODU4IDIuODU4IDIuMjktMi4yODkgMi4yOSAyLjI4OSAyLjg1OC0yLjg1OHptLTQuNTgtMi44NTlsMi4yOS0yLjI4OCA0LjE0My00LjE0My0yLjg1OC0yLjg1OC0yLjI5IDIuMjg5LTIuMjktMi4yODktMi44NTggMi44NTh6Ii8+PC9zdmc+)

# 📚🤖 Conversational RAG: PDF Q&A ChatBot

⚡ **A Next-Gen Document Understanding Assistant** ⚡

Transform your PDF documents into interactive knowledge bases! This AI-powered chatbot combines:

🔍 **Semantic Search**  
💡 **Contextual Understanding**  
🚀 **Lightning-Fast Responses**

Built with cutting-edge RAG architecture powered by GROQ's accelerated AI platform and Hugging Face's embeddings.

## 🚀 Features

<div align="center">

✨ **Core Capabilities** ✨

</div>

|   |   |
|---|---|
| 📤 **Multi-PDF Upload** | Simultaneously process multiple documents |
| 🧠 **Smart Chunking** | Intelligent text splitting with configurable overlap |
| 🔎 **Contextual Search** | Find relevant passages using FAISS vector store |
| 💬 **Natural Q&A** | Human-like conversations powered by Llama3-8b |
| ⏱ **Performance Metrics** | Response time tracking for optimization |
| 🛠 **Customizable** | Adjust chunk sizes, models, and prompts |

## 🛠 Technology Stack

| Component              | Technology                          |
|------------------------|-------------------------------------|
| **Large Language Model** | ![GROQ](https://img.shields.io/badge/GROQ-Llama3--8b-01B4A4?logo=groq) |
| **Embeddings**         | ![HuggingFace](https://img.shields.io/badge/HuggingFace-MiniLM--L6v2-yellow?logo=huggingface) |
| **Vector Store**       | ![FAISS](https://img.shields.io/badge/Facebook-FAISS-4267B2?logo=facebook) |
| **Document Processing** | ![LangChain](https://img.shields.io/badge/LangChain-Text%20Splitting-blue?logo=langchain) |
| **UI Framework**       | ![Streamlit](https://img.shields.io/badge/Streamlit-Web%20UI-FF4B4B?logo=streamlit) |

## Prerequisites

- Python 3.8+
- GROQ API key
- HuggingFace token

## Installation

1. Clone the repository

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   HF_TOKEN=your_huggingface_token_here
   ```

## Usage

1. Start the application:
   ```
   streamlit run RAG_PDFbot_with_GROQAPI.py
   ```

2. The application will open in your default web browser.

### User Interface Flow

#### Landing Page

![Landing Page](output_images/landing.png)

The landing page displays the title of the application and provides a file uploader for PDF documents.

#### File Upload Failed

![Upload Failed](output_images/uploadsfail.png)

If you attempt to submit without uploading any PDF files, the application will display a warning message.

#### Vector Embeddings Creation

![Submit PDFs](output_images/submit.png)

After uploading PDF files and clicking the "Submit PDFs" button, the application processes the documents and creates vector embeddings. A success message is displayed when the embeddings are ready.

#### Question Answering

![Final Q&A Interface](output_images/final.png)

Once the vector embeddings are created, you can enter questions about the content of the uploaded PDFs. The application will retrieve relevant information from the documents and provide an answer along with the response time.

## How It Works

1. **Document Processing**:
   - The application loads PDF documents using PyPDFLoader
   - Documents are split into smaller chunks using RecursiveCharacterTextSplitter
   - Text chunks are converted into vector embeddings using HuggingFace's embedding model
   - Embeddings are stored in a FAISS vector database for efficient similarity search

2. **Question Answering**:
   - When a user asks a question, the application searches for the most relevant document chunks
   - The retrieved chunks are sent as context to the GROQ LLM (Llama3-8b-8192)
   - The LLM generates an answer based on the provided context
   - The answer and response time are displayed to the user

## Code Structure

- **Environment Setup**: Loading API keys from .env file
- **LLM Initialization**: Setting up the GROQ LLM with the Llama3 model
- **Prompt Template**: Defining how to format the context and question for the LLM
- **UI Components**: Streamlit interface elements for file upload and user interaction
- **Vector Embedding Creation**: Processing PDFs and creating searchable embeddings
- **Retrieval Chain**: Connecting the vector store to the LLM for context-based answers

## Customization

You can customize the application by modifying the following parameters:

- **Chunk Size and Overlap**: Adjust the text splitting parameters for different document types
  ```python
  st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  ```

- **LLM Model**: Change the GROQ model to use different versions
  ```python
  llm = ChatGroq(
      groq_api_key=groq_api_key,
      model_name="Llama3-8b-8192"
  )
  ```
- **Prompt Template**: Modify the prompt to change how the LLM responds


## License

This project is open source and available under the [GNU General Public License v3.0 (GPL-3.0)](LICENSE).

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the document processing and retrieval framework
- [GROQ](https://groq.com/) for the LLM API
- [HuggingFace](https://huggingface.co/) for the embedding models
- [FAISS](https://github.com/facebookresearch/faiss) for the vector database
- [Streamlit](https://streamlit.io/) for the user interface
