import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import AgentExecutor, initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from time import sleep
from typing import List
import time
from dotenv import load_dotenv 
# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize session state
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def create_semantic_chunks(text: str, chunk_size: int = 512, overlap: int = 50) -> List[Document]:
    """
    Creates semantically meaningful chunks using a hybrid approach of recursive splitting
    and content-based segmentation
    """
    # First, split into major semantic units (e.g., paragraphs)
    initial_splits = text.split('\n\n')
    
    # Create a more sophisticated text splitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""],
        is_separator_regex=False
    )
    
    documents = []
    
    for i, section in enumerate(initial_splits):
        if not section.strip():
            continue
            
        try:
            # Further split large sections while preserving semantic meaning
            if len(section) > chunk_size:
                chunks = splitter.split_text(section)
                for chunk in chunks:
                    if chunk.strip():  # Only add non-empty chunks
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": f"section_{i}",
                                "chunk_type": "split_chunk"
                            }
                        ))
            else:
                # Keep smaller sections intact to preserve context
                documents.append(Document(
                    page_content=section,
                    metadata={
                        "source": f"section_{i}",
                        "chunk_type": "full_section"
                    }
                ))
                
        except Exception as e:
            st.warning(f"Error processing section {i}: {str(e)}")
            continue
            
        # Add small delay to avoid rate limiting
        time.sleep(0.1)
    
    return documents

def get_vector_store(documents: List[Document]):
    """Create and save vector store from documents"""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if not documents:
        return None
    
    try:
        # Add retry mechanism for vector store creation
        max_retries = 3
        for attempt in range(max_retries):
            try:
                vector_store = FAISS.from_documents(
                    documents,
                    embedding=embeddings
                )
                vector_store.save_local("faiss_index")
                st.session_state.pdf_processed = True
                return vector_store
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)  # Exponential backoff
                
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None

def process_documents(text: str) -> FAISS:
    """Process documents using semantic chunking and create vector store"""
    # Create semantic chunks
    documents = create_semantic_chunks(text)
    
    if not documents:
        st.error("No valid chunks were created from the documents")
        return None
    
    # Create and return vector store
    return get_vector_store(documents)

def search_pdfs(query: str) -> str:
    """Search through uploaded PDFs for relevant information."""
    if not st.session_state.pdf_processed:
        return "NO_PDF_AVAILABLE"
        
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if not os.path.exists("faiss_index"):
            return "NO_PDF_AVAILABLE"
            
        pdf_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = pdf_db.similarity_search(query, k=3)
        
        if not docs:
            return "NO_RELEVANT_INFO"
        
        context = "\n\n".join(f"From PDF Document:\n{doc.page_content}" 
                            for doc in docs)
        return context
            
    except Exception as e:
        return f"ERROR: {str(e)}"

def process_web_content(loader: WebBaseLoader) -> List[Document]:
    """Process web content using semantic chunking"""
    docs = loader.load()
    text = "\n\n".join([doc.page_content for doc in docs])
    return create_semantic_chunks(text)

def create_tools():
    tools = []
    
    # Wikipedia Tool
    wiki = WikipediaAPIWrapper(
        top_k_results=2,
        doc_content_chars_max=500,
        load_all_available_meta=True
    )
    wiki_tool = WikipediaQueryRun(
        api_wrapper=wiki,
        name="wikipedia",
        description="Useful for questions about general knowledge, historical facts, and encyclopedia-like information."
    )
    tools.append(wiki_tool)
    
    # Arxiv Tool
    arxiv = ArxivAPIWrapper(
        top_k_results=2,
        load_all_available_meta=True,
        sort_by="relevancy"
    )
    arxiv_tool = ArxivQueryRun(
        api_wrapper=arxiv,
        name="arxiv",
        description="Useful for questions about scientific papers, research, and academic content."
    )
    tools.append(arxiv_tool)
    
    # Web Tool
    try:
        loader = WebBaseLoader(
            "https://www.langchain.com/langsmith",
            verify_ssl=False,
            requests_kwargs={"timeout": 10}
        )
        documents = process_web_content(loader)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        web_tool = create_retriever_tool(
            retriever,
            "web_search",
            "Search for information from web documentation. Use this for technical questions."
        )
        tools.append(web_tool)
    except Exception as e:
        st.warning(f"Web tool creation failed: {str(e)}")
    
    # News Tool
    try:
        urls = [
            "https://news.google.com/home?hl=en-IN&gl=IN&ceid=IN:en",
        ]
        
        loader = WebBaseLoader(
            urls,
            verify_ssl=False,
            requests_kwargs={"timeout": 10}
        )
        documents = process_web_content(loader)
        
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vectordb = FAISS.from_documents(documents, embeddings)
        retriever = vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
        news_tool = create_retriever_tool(
            retriever,
            "news_search",
            "Search for recent news, Top news, trends, real-time updates. Use this for News and real-time knowledge"
        )
        tools.append(news_tool)
    except Exception as e:
        st.warning(f"News tool creation failed: {str(e)}")
    
    # PDF Tool
    pdf_tool = Tool(
        name="pdf_search",
        func=search_pdfs,
        description="""Use this tool to search within uploaded PDF documents. 
                      If the response is 'NO_PDF_AVAILABLE', it means no PDFs have been uploaded yet.
                      If the response is 'NO_RELEVANT_INFO', it means no relevant information was found in the PDFs."""
    )
    tools.append(pdf_tool)
    
    return tools

def get_conversational_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.5,
        top_p=0.9,
        top_k=40,
        max_output_tokens=2048
    )
    
    tools = create_tools()
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    system_message = """You are a helpful AI assistant that can search through multiple sources including PDFs, Wikipedia, Arxiv, and web documentation.
    When using the pdf_search tool:
    - If you receive 'NO_PDF_AVAILABLE', inform the user that no PDFs have been uploaded yet and they should upload and process PDFs first.
    - If you receive 'NO_RELEVANT_INFO', inform the user that no relevant information was found in the uploaded PDFs.
    - If you receive actual content, incorporate it into your response and cite it as coming from the PDF.
    
    Always try to provide comprehensive answers by combining information from multiple sources when appropriate."""
    
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        verbose=True,
        max_iterations=3,
        early_stopping_method="generate",
        handle_parsing_errors=True,
        system_message=system_message
    )
    
    return agent_executor

def handle_user_input(user_question, agent_executor):
    try:
        if os.path.exists("faiss_index"):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            pdf_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        response = agent_executor.invoke({
            "input": user_question
        })
        
        if "output" in response:
            st.write("Answer:", response["output"])
        else:
            st.error("No response generated")
    except Exception as e:
        st.error(f"Error processing question: {str(e)}")

def main():
    st.set_page_config(page_title="Multi-source RAG Assistant")
    st.header("Interactive RAG-based LLM System")
    
    if "agent_executor" not in st.session_state:
        st.session_state.agent_executor = get_conversational_chain()
    
    with st.sidebar:
        st.title("Document Upload")
        pdf_docs = st.file_uploader(
            "Upload PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )
        
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    try:
                        raw_text = get_pdf_text(pdf_docs)
                        vector_store = process_documents(raw_text)
                        if vector_store:
                            st.success("Documents processed successfully!")
                            st.session_state.agent_executor = get_conversational_chain()
                        else:
                            st.error("Error processing documents")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please upload PDF files first")
    
    st.write("PDF Status:", "PDFs processed and ready" if st.session_state.pdf_processed else "No PDFs processed yet")
    user_question = st.text_input("Ask your question:")
    if user_question:
        handle_user_input(user_question, st.session_state.agent_executor)

if __name__ == "__main__":
    main()