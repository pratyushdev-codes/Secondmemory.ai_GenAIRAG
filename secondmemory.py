import os
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
import google.generativeai as genai
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import WebBaseLoader

# Load environment variables
load_dotenv()

# Set Google API key for Gemini
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    print("Google API key has been set.")
else:
    print("Error: Google API key not found in .env file.")

# Define tools
api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# Load and process PDF
document_loader = PyPDFLoader('./Datasets/attention.pdf')
documents = document_loader.load()

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    task_type="retrieval_document"
)
faiss_db = FAISS.from_documents(chunks, embedding=embeddings)
retriever = faiss_db.as_retriever()

# Create retriever tool
retriever_tool = create_retriever_tool(
    retriever,
    name="document_search",
    description="Search for information in the PDF document"
)

# Define LLM and QA chain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)
chain = load_qa_chain(llm, chain_type="stuff")

# Define prompt template
prompt_template = """
Answer the questions asked based on the context provided by the retriever. If you do not find any relevant information in the context provided, reply that you couldn't find a relevant answer in the context.
<context>
{context}
</context>

Question: {input}
"""

# Create retrieval chain
chat_prompt_template = ChatPromptTemplate.from_template(prompt_template)
stuff_document_chain = create_stuff_documents_chain(llm, chat_prompt_template)
retrieval_chain = create_retrieval_chain(retriever, stuff_document_chain)

# Define tools list
tools = [wiki_tool, arxiv_tool, retriever_tool]

# Example usage
query ="Summarize the document"
response = retrieval_chain.invoke({'input': query})
print(response)