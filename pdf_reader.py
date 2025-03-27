import streamlit as st
import io
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")

genai.configure(api_key=API_KEY)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(io.BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            extracted_text = page.extract_text() or ""
            text.append(extracted_text)
    return "\n".join(text)

# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create and store embeddings
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state["vector_store"] = vector_store

# Function to set up a conversational chain
def get_conversational_chain():
    prompt_template = PromptTemplate(
        template="""Answer the question as detailed as possible from the provided context.
        If the answer is not in the provided context, say "Answer is not available in the context."
        
        Context:\n {context}\n
        Question: {question}\n
        Answer:\n""",
        input_variables=["context", "question"]
    )
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
    return load_qa_chain(model, chain_type="stuff", prompt=prompt_template)

# Function to handle different query methods
def query_documents(user_question, query_method):
    if "vector_store" not in st.session_state:
        st.warning("Please upload and process PDF files first.")
        return "Please upload and process PDFs first."

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Perform search based on query method
    if query_method == "Semantic Search":
        docs = new_db.similarity_search(user_question)
    elif query_method == "Keyword Search":
        results = new_db.similarity_search_with_score(user_question)
        docs = [doc for doc, score in results if isinstance(doc, Document)]  # Ensure only Documents
    else:  # Full-Text Search
        docs = new_db.max_marginal_relevance_search(user_question)

    # Ensure docs only contain valid LangChain Document objects
    docs = [doc for doc in docs if hasattr(doc, "page_content")]

    if not docs:
        return "No relevant documents found. Try rewording your question."

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response.get("output_text", "No response available.")

# Streamlit UI
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("Chat with Multiple PDFs using Gemini AI")

    # Dark mode toggle
    dark_mode = st.sidebar.checkbox("Dark Mode")
    if dark_mode:
        st.markdown("<style>body { background-color: #0E1117; color: white; }</style>", unsafe_allow_html=True)

    user_question = st.text_input("Ask a question from the uploaded PDF files:")
    query_method = st.radio("Select Query Method", ["Semantic Search", "Keyword Search", "Full-text Search"], index=0)

    if user_question:
        with st.chat_message("User"):
            st.write(user_question)

        response = query_documents(user_question, query_method)

        with st.chat_message("Bot"):
            st.write(response)

    with st.sidebar:
        st.title("Menu")
        pdf_docs = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
                return
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Processing complete! You can now ask questions.")

                # Show extracted text preview
                st.subheader("Extracted Text Preview")
                st.text_area("Extracted Content", raw_text[:1000] + "...", height=300)

        if st.button("Reset Session"):
            st.session_state.clear()
            st.success("Session reset successfully!")

if __name__ == "__main__":
    main()
