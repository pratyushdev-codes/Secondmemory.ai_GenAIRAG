from crewai import Task
from tools import yt_extractor_tool
from agents import yt_agent_researcher,yt_agent_writer, file_read_tool, pdf_agent_researcher


##------------------ Tasks ------------------



##------------------ 1. Extract information from YT Videos ------------------

research_Taks=Task(
    description=(
        "Identify the Subject and the context of the Youtube Videos {query}." 
        "This task is to research about the Youtube videos and extract the information from the videos."
    ),
    expected_output="Search for information in the Youtube Video. A comprehensive 1 paragraph summary with metrics of the video with respect to the search query",
    toots=[yt_extractor_tool],
    agent=yt_agent_researcher,
)

write_task=Task(
    description=(
        "Get the information extracted from the Youtube Videos {query}and respond to the user query with that information."  
         ),
    expected_output="Respond to the user queries extracted from Youtube videos.",
    tools=[yt_extractor_tool],
    agent=yt_agent_writer,
    async_execution=False,
    output_json=True,   
)

##------------------ 2. Extract information from PDFs ------------------

query_pdf=Task(
    description=(
        "Extract Answers for the  {query} asked based on the data provided in the PDF. If you do not find any relevant information in the context provided, reply that you couldn't find a relevant answer in the context." 
        "This task is to research about the PDF and extract the information from the PDF."
    ),
    expected_output="Search for information in the PDF. A comprehensive 1 paragraph summary with metrics of the PDF with respect to the search query",
    toots=[file_read_tool],
    agent=pdf_agent_researcher,
    async_execution=False,
    output_json=True,
)

##------------------ 3. Extract information from Websites ------------------

query_web=Task(
    description=(
        "Extract Answers for the {query} asked based on the data provided in the Website URL. If you do not find any relevant information in the context provided, reply that you couldn't find a relevant answer in the context." 
        "This task is to research about the PDF and extract the information from the Website URL."
    ),
    expected_output="Search for information in the Website. A comprehensive 1 paragraph summary with metrics of the PDF with respect to the search query",
    toots=[file_read_tool],
    agent=pdf_agent_researcher,
    async_execution=False,
    output_json=True,
)

##------------------ 4. Extract information from Documents ------------------