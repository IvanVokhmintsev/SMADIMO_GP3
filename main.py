from langchain.agents import create_agent

from dotenv import load_dotenv
from PipelineMemory import PipelineMemory
from prompts import AGENT_ONE_SYS_PROMPT, AGENT_TWO_SYS_PROMPT, AGENT_TWO_USER_PROMPT, AGENT_ONE_USER_PROMPT
from tools.download_file import make_download_tools
from tools.make_eda_tools import make_eda_tools
from tools.make_fe_tools import make_fe_tools
from tools.ml_tools import make_ml_tools

load_dotenv()

state = PipelineMemory()

download_tools = make_download_tools(state)
eda_tools = make_eda_tools(state)
feature_tools = make_fe_tools(state)
ml_tools = make_ml_tools(state)

eda_agent = create_agent(
    model="google_genai:gemma-4-31b-it",
    tools=download_tools + eda_tools + feature_tools,
    system_prompt=AGENT_ONE_SYS_PROMPT
)

ml_agent = create_agent(
    model="google_genai:gemma-4-31b-it",
    tools=ml_tools,
    system_prompt=AGENT_TWO_SYS_PROMPT
)

result = eda_agent.invoke(
    {"messages": [
        {"role": "user",
         "content": AGENT_ONE_USER_PROMPT}]}
)


print(result["messages"][-1].content_blocks)

print(state.df.head())
print(state.df.info())

result = ml_agent.invoke(
    {"messages": [
        {"role": "user",
         "content":
             AGENT_TWO_USER_PROMPT}]}
)