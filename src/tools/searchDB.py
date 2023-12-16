import os

import openai
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType, initialize_agent, tool
import langchain
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.tools import tool

import json
import requests
import datetime
from pydantic.v1 import BaseModel, Field

# llmモデルの初期化
llm = AzureChatOpenAI( # Azure OpenAIのAPIを読み込み。
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
)


# システムプロンプトの設定
SEARCHDB_SYSTEM_PROMPT = '''あなたはデータベース検索AIです。
特定のデータベースを操作し、検索した結果を返答します。

あなたの取るべき行動
--------------------
ユーザーから与えられたプロンプトから検索ワードを抽出し、 search_word という変数に格納してください。
検索結果は serach_result という変数に格納されています。
search_result に検索結果が格納されている場合は、検索結果を元に回答を作成し、ユーザーに返答してください。
最終回答は150字以下に要約して回答してください。
もし検索ワードに対する検索結果が不適当であれば、答えを作ろうとせず "わかりません。" と回答し、ユーザーに対して検索ワードを変更するように促してください。
--------------------

'''


# エージェントの初期化
class SearchInput(BaseModel) : # 検索ワードを入力するためのモデルを作成。
    search_word: str = Field(description="ユーザーからの入力から生成される検索ワードです。")

@tool("search", args_schema=SearchInput) # Agentsツールを作成。
def search(
    search_word: str,
):
    """検索ワードから、検索結果を返答します。"""
    search_word = search_word
    serach_result = None
    
    def search_database(search_word):
        retriever = AzureCognitiveSearchRetriever(
            service_name=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
            index_name=os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"],
            api_key=os.environ["AZURE_SEARCH_KEY"],
            content_key="content",
            top_k=3
        )
        res = retriever.get_relevant_documents(query=search_word)
        i = 1
        search_result = []
        for doc in res:
            if hasattr(doc, 'page_content'):
                search_result.append(f'検索結果{i}は以下の通りです。\n{doc.page_content}')
                i += 1
        return search_result

    serach_result = search_database(search_word)
    return serach_result

verbose = True
langchain.debug = verbose

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
chat_history = MessagesPlaceholder(variable_name='chat_history')

search_tools = [search]

agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(template=SEARCHDB_SYSTEM_PROMPT),
    "extra_prompt_messages": [chat_history]
}
search_agent = initialize_agent(
    search_tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=verbose,
    agent_kwargs=agent_kwargs,
    memory=memory
)


while True:
    message = input(">> ")
    if message == "exit" or message == ":q":
        break
    try:
        search_agent.run(message)
    except openai.error.InvalidRequestError as e:
        print(e)

