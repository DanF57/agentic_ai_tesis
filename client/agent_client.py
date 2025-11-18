# client/agent_client.py
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage
from haystack_integrations.tools.mcp import MCPToolset, StreamableHttpServerInfo
from client.utils.llm_factory import create_llm
from dotenv import load_dotenv
import yaml

PROVIDER = "openai"

load_dotenv()
with open("config/models.yaml", 'r') as f:
    config = yaml.safe_load(f)
    provider_config = config['providers'][PROVIDER]
    model_name = provider_config['model']

# Create MCP toolset pointing to your server
server_info = StreamableHttpServerInfo(url="http://localhost:8000/mcp")
toolset = MCPToolset(
    server_info=server_info,
    # tool_names=["search_web_mock"]  # Specify tools or omit to load all
)

# Create LLM
chat_generator = create_llm(
    provider=PROVIDER,
    model=model_name,
)

# Create agent with toolset
agent = Agent(
    chat_generator=chat_generator,
    system_prompt="""You're a helpful agent for college students. When asked about information, 
                     use the search_web_mock tool to find the information and then summarize the findings.
                     When you get web search results, extract the relevant information and present it in a clear, 
                     concise manner.""",
    tools=toolset
)

# Run agent
user_message = ChatMessage.from_user("Answer this 2 questions: 1.What is your role and what tools you have? 2.How many steps does EDA have and according to who?")
response = agent.run(messages=[user_message])
print("Respuesta del modelo:\n", response["messages"][-1].text)




