from crewai import Agent
from tools import yt_extractor_tool, file_read_tool, pdf_extractor_tool ,web_search_tool, google_search_tool

##------------------ Imports ------------------
from crewai import LLM


##Importiing .env file for Gemini LLM
from dotenv import load_dotenv
import os
load_dotenv()


# ## Calling Gemoni Gen AI Model
# import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI

## Defining LLM
# llm =LLM(
#     model="gemini/gemini-1.5-flash",
#     temperature=0.3,
#     # verbose=True,
#     google_api_key=os.getenv("GOOGLE_API_KEY")
# )

llm = LLM(
    model="gemini/gemini-1.5-pro-latest",
    temperature=0.7
)

 ## Creating an agent for scraping Youtube videos and extracting the video information
 
yt_agent_researcher = Agent(
     role="Agents for extracting information from Youtube videos",
     goal="To extract information from Youtube videos {query}",
     verbose=True,
     memory =True,
     backstory=(
         "This Agent is an expert in understanding videos from Youtube and extracting information from them and server to user in a chatbot interface."
     ),
     llm=llm,
     tools=[yt_extractor_tool],
     allow_delegation=True,
 )


## writing content extracted from Youtube videos

yt_agent_writer = Agent(
    role="Writers for Youtube videos",
    goal="To write content extracted from Youtube videos {query}",
    verbose=True,
    memory=True,
    backstory=(
        "This Agent is an expert in writing content extracted from Youtube videos."
    ),
    llm=llm,
    tools=[yt_extractor_tool],
    allow_delegation=False,
)


## Agent for Scraping pdfs and extracting information from them
pdf_searcher_agent = Agent(
    role="Search User Query in PDF",
    goal="To search for user query in the  content extracted from PDF {query}",
    verbose=True,
    memory=True,
    backstory=(
        "This Agent is an expert in searching content and answering questions from PDF {query}."
    ),
    llm=llm,
    tools=[pdf_extractor_tool],
    allow_delegation=True,
)


##Agent for Searching websites from their URL

web_searcher_agent = Agent(
    role="Search User Query in Website",
    goal="To search for user query from Website URL {query}",
    verbose=True,
    memory=True,
    backstory=(
        "This Agent is an expert in searching content and answering questions from Websites ."
    ),
    llm=llm,
    tools=[web_search_tool],
    allow_delegation=True,
)

## Agent for searching query on Google with Serper API 
google_searcher_agent = Agent(
    role="Search User Query in Google",
    goal="To search for user query in Google {query}",
    verbose=True,
    memory=True,
    backstory=(
        "This Agent is an expert in searching content and answering questions from Google."
    ),
    llm=llm,
    tools=[google_search_tool],
    allow_delegation=True,
)



