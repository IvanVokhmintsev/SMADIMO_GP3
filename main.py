from langchain.agents import create_agent

from dotenv import load_dotenv
from PipelineMemory import PipelineMemory
from tools.download_file import make_download_tools
from tools.make_eda_tools import make_eda_tools

load_dotenv()

state = PipelineMemory()

download_tools = make_download_tools(state)
eda_tools = make_eda_tools(state)


agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    tools=download_tools,
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "Скачай файл по данному url https://drive.usercontent.google.com/u/0/uc?id=1PDpIeIhzFJnEtaSUZmtZPM2pq8FxvfhF&export=download"}]}
)
print(result["messages"][-1].content_blocks)