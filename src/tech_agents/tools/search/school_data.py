import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import AgentType, tool
from langchain_community.retrievers import AzureCognitiveSearchRetriever
from pydantic.v1 import BaseModel, Field

from ...template.agent_model import BaseToolAgent


# システムプロンプトの設定
# SEARCHDB_SYSTEM_PROMPT = '''あなたはデータベース検索AIです。
# 特定のデータベースを操作し、検索した結果を返答します。

# あなたの取るべき行動
# --------------------
# ユーザーから与えられたプロンプトから検索ワードを抽出し、 search_word という変数に格納してください。
# 検索結果を元に回答を作成し、ユーザーに返答してください。
# 最終回答は150字以下に要約して回答してください。
# もし検索ワードに対する検索結果が不適当であれば、答えを作ろうとせず "わかりません。" と回答し、ユーザーに対して検索ワードを変更するように促してください。
# --------------------

# 例
# --------------------
# 例えば、ユーザーから "京都テックについて教えて" というプロンプトを受け取った場合、 searach_word に "京都テック" を格納します。
# その後、"京都テック" という検索ワードを元に検索を行い、検索結果を元に回答を作成し、ユーザーに返答します。
# --------------------

# '''
SEARCHDB_SYSTEM_PROMPT = '''You are a database search AI. You operate a specific database and return search results as responses.
Respond in Japanese.

# Your actions
--------------------
Extract search keywords from the prompts given by the user and store them in a variable named 'search_word'.
Create a response based on the search results and reply to the user.
If the search results for the given keyword are inappropriate, do not attempt to formulate an answer. Instead, respond with 'I don't know.' and prompt the user to change the search keyword.
--------------------

# Example
--------------------
For instance, if you receive a prompt from the user saying 'Tell me about Kyoto Tech,' store 'Kyoto Tech' as the search_word.
Then, conduct a search using the keyword 'Kyoto Tech,' create a response based on the search results, and reply to the user.
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
                search_result.append(f'・検索結果{i}は以下の通りです。\n{doc.page_content}\n\n')
                i += 1
        return search_result

    serach_result = search_database(search_word)
    return serach_result


search_tools = [search]



class SchoolAgentInput(BaseModel):
    user_utterance: str = Field(
        description="The user's most recent utterance that is communicated to the person in charge of the school database search.")


class SchoolAgent(BaseToolAgent):
    def __init__(self, llm, memory, chat_history, verbose):
        super().__init__(llm, memory, chat_history, verbose)
        # SchoolAgent 特有の初期化（もしあれば）

    def run(self, input):
        # SchoolAgent特有の処理
        school_agent = self.initialize_agent(
            agent_type=AgentType.OPENAI_FUNCTIONS,
            tools=search_tools,  # 事前に定義されたsearch関数
            system_message_template=SEARCHDB_SYSTEM_PROMPT
        )
        return school_agent.run(input)

