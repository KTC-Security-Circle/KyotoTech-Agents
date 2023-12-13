import os

# OpenAI のラッパーをインポート
from langchain.tools import BaseTool
from langchain import SerpAPIWrapper, LLMMathChain
from langchain.llms import OpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.chat_models import AzureChatOpenAI
# LLM ラッパーを初期化
llm = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
)


# DuckDuckGoSearchRun のラッパーを初期化
search = DuckDuckGoSearchRun()

# BaseTool クラスのインポート

# 検索を行うツールを定義
class CustomSearchTool(BaseTool):
    name = "duckduckgo-search"
    description = "useful for when you need to search for latest information in web"

    def _run(self, query: str) -> str:
        """Use the tool."""
        return search.run(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
