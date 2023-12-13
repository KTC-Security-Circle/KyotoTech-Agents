from dotenv import load_dotenv
load_dotenv(override=True)
import os

# https://zenn.dev/umi_mori/books/prompt-engineer/viewer/langchain_agents からの引用
from langchain.llms import OpenAI
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import AzureChatOpenAI

llm = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
)

# tool_names = ["llm-math"]
# tools = load_tools(tool_names, llm=llm)

# search = DuckDuckGoSearchRun()
# tools.append(
#     Tool(
#         name="duckduckgo-search",
#         func=search.run,
#         description="useful for when you need to search for latest information in web"
#     )
# )

# ツールの一覧を定義
from tools.classes import CustomSearchTool
tools = [
    CustomSearchTool()
]

agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
agent.run("""
現在の20代の日本人男性の平均身長を教えて。
そして、私の身長は168cmなため、日本全国から見た時の差を2乗した結果を教えて。
""")