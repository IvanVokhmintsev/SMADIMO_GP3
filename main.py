from langchain.agents import create_agent

from dotenv import load_dotenv
from PipelineMemory import PipelineMemory
from prompts import AGENT_ONE_SYS_PROPMT
from tools.download_file import make_download_tools
from tools.make_eda_tools import make_eda_tools

load_dotenv()

state = PipelineMemory()

download_tools = make_download_tools(state)
eda_tools = make_eda_tools(state)


agent = create_agent(
    model="google_genai:gemma-4-31b-it",
    tools=download_tools + eda_tools,
    system_prompt=AGENT_ONE_SYS_PROPMT
)


URL = 'https://drive.usercontent.google.com/download?id=1LEwPH9AQw3q-x6HM_e9amRkB2tcnho_n&export=download&authuser=0'
result = agent.invoke(
    {"messages": [
        {"role": "user",
         "content":
             f"Скачай файл по данному url {URL} и сделай EDA. Датасет представляет из себя данные о зарплатах с LinkedIn. Мы хотели бы предиктить зарплату."}]}
)

print(result["messages"][-1].content_blocks)

print(state.df.head())
print(state.df.info())