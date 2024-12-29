from crewai import Crew , Process
from agents import yt_agent_researcher, yt_agent_writer, pdf_searcher_agent, web_searcher_agent
from tools import yt_extractor_tool, file_read_tool, web_search_tool, google_search_tool
from tasks import research_Taks, write_task, query_pdf, query_web




##initialize the Crew

engine = Crew(
    agents = [yt_agent_researcher, yt_agent_writer, pdf_searcher_agent, web_searcher_agent],
    tasks = [research_Taks, write_task, query_pdf, query_web],
    process = Process.sequential,
    memory = True,
    cache = True,
    max_rpm = 100,
    share_crew = True, 
)

##Start the execution process
result = Crew.kickOff(inputs={"query":"Explain me the new Advances in AI"})

print(result);

