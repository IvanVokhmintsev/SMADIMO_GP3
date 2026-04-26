from langchain.agents import create_agent

from dotenv import load_dotenv
from PipelineMemory import PipelineMemory
from tools.make_eda_tools import make_eda_tools

load_dotenv()

state = PipelineMemory()

eda_tools = make_eda_tools(state)

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

agent = create_agent(
    model="google_genai:gemini-2.5-flash-lite",
    tools=[get_weather],
    system_prompt="You are a helpful assistant",
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather in San Francisco?"}]}
)
print(result["messages"][-1].content_blocks)