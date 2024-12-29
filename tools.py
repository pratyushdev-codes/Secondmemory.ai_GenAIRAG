from crewai_tools import YoutubeVideoSearchTool
## youtube search tool

from crewai_tools import FileReadTool
##PDF READER TOOLS

from crewai_tools import WebsiteSearchTool
## Website Search Tool

from crewai_tools import SerperDevTool
## google search tool

from dotenv import load_dotenv
import os

## intializing YT Tools
yt_extractor_tool = YoutubeVideoSearchTool(youtube_video_url='https://www.youtube.com/watch?v=UV81LAb3x2g')

## Initializing PDF Tools
# Initialize the tool to read any files the agents knows or lean the path for
file_read_tool = FileReadTool()
file_read_tool = FileReadTool(file_path='./Datasets/attention.pdf')


##Intializing tools for Searching websites with the help of the URL

web_search_tool = WebsiteSearchTool()
tool = WebsiteSearchTool(website='https://example.com')



## Tools for searching on Google
load_dotenv()
os.eviron['SERPER_API_KEY']= os.getenv('SERPER_API_KEY')


tool = SerperDevTool()

# print(tool.run(search_query="ChatGPT"))


    # search_url="https://google.serper.dev/scholar",
    # n_results=2,