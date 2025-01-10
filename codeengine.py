# Save this file as "app.py"

import streamlit as st
import os
from dotenv import load_dotenv
from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

def clone_repository(repo_url: str, target_dir: str) -> None:
    """Clone a git repository to the specified directory."""
    if not os.path.exists(target_dir):
        os.system(f"git clone {repo_url} {target_dir}")
    else:
        print(f"Directory {target_dir} already exists. Skipping clone.")

def convert_files_to_txt(src_dir: str, dst_dir: str) -> None:
    """Convert repository files to txt format while preserving directory structure."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    dst_path.mkdir(parents=True, exist_ok=True)

    for file_path in src_path.rglob('*'):
        if file_path.is_file() and not file_path.suffix == '.jpg':
            relative_path = file_path.relative_to(src_path)
            new_path = dst_path / relative_path.parent / (relative_path.name + '.txt')
            new_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                content = file_path.read_text(encoding='utf-8')
                new_path.write_text(content, encoding='utf-8')
            except UnicodeDecodeError:
                try:
                    content = file_path.read_text(encoding='latin-1')
                    new_path.write_text(content, encoding='utf-8')
                except UnicodeDecodeError:
                    print(f"Failed to decode file: {file_path}")

def process_documents(src_dir: str) -> List:
    """Load and process documents from the source directory."""
    loader = DirectoryLoader(
        src_dir,
        glob="**/*.txt",
        show_progress=True,
        loader_cls=TextLoader
    )
    documents = loader.load()
    st.write(f"Number of files loaded: {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    split_documents = text_splitter.split_documents(documents=documents)
    st.write(f"Number of chunks: {len(split_documents)}")

    for doc in split_documents:
        doc.metadata["source"] = doc.metadata["source"].replace(".txt", "")

    return split_documents

def initialize_models():
    """Initialize the LLM and embeddings models."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    return llm, embeddings

def create_vector_store(documents: List, embeddings, store_path: str) -> Qdrant:
    """Create and populate the vector store."""
    return Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path=store_path,
        collection_name="my_documents"
    )

def create_qa_chain(llm, vector_store: Qdrant):
    """Create the question-answering chain."""
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True
    )

# Streamlit app
st.set_page_config(page_title="Codebase Q&A System", layout="wide")

st.title("Codebase Q&A System")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
    st.session_state.qa_chain = None

# Initialize button
if not st.session_state.initialized:
    if st.button("Initialize System"):
        with st.spinner("Initializing the system..."):
            try:
                # Configuration
                REPO_URL = "https://github.com/underratedcoderr/trail"
                REPO_DIR = "codebase"
                CONVERTED_DIR = "converted_codebase"
                VECTOR_STORE_PATH = "local_qdrant"

                # Clone repository
                clone_repository(REPO_URL, REPO_DIR)

                # Convert files to txt
                convert_files_to_txt(REPO_DIR, CONVERTED_DIR)

                # Process documents
                documents = process_documents(CONVERTED_DIR)

                # Initialize models
                llm, embeddings = initialize_models()

                # Create vector store
                vector_store = create_vector_store(documents, embeddings, VECTOR_STORE_PATH)

                # Create QA chain
                st.session_state.qa_chain = create_qa_chain(llm, vector_store)
                st.session_state.initialized = True
                
                st.success("System initialized successfully!")
            except Exception as e:
                st.error(f"Error during initialization: {str(e)}")

# Query interface
if st.session_state.initialized:
    query = st.text_input("Enter your question about the codebase:")
    
    if query:
        with st.spinner("Searching for answer..."):
            try:
                response = st.session_state.qa_chain.invoke(query)
                
                st.write("### Answer:")
                st.write(response['result'])
                
                st.write("### Source Documents:")
                for doc in response['source_documents']:
                    with st.expander(f"Source: {doc.metadata['source']}"):
                        st.write(doc.page_content)
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
else:
    st.info("Please initialize the system first.")