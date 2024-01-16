from pydantic.v1 import BaseModel, Field

from langchain.agents import AgentType, tool

from ...db.vector import search_vector
from ...template.agent_model import BaseToolAgent


# システムプロンプトの設定
# SEARCHDB_SYSTEM_PROMPT = '''あなたはデータベース検索AIです。
# 特定のデータベースを操作し、検索した結果を返答します。

# あなたの取るべき行動
# --------------------
# ユーザーから与えられたプロンプトから検索ワードを抽出し、 search_word という変数に格納してください。
# 検索結果を元に回答を作成し、ユーザーに返答してください。
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


class SearchInput(BaseModel):  # 検索ワードを入力するためのモデルを作成。
    search_word: str = Field(description="ユーザーからの入力から生成される検索ワードです。")


@tool("search", args_schema=SearchInput)  # Agentsツールを作成。
def search(
    search_word: str,
):
    """検索ワードから、検索結果を返答します。"""
    def search_database(search_word):
        docs = search_vector("vector-class-data", search_word)
        i = 1
        search_result = []
        for doc in docs:
            if hasattr(doc, 'metadata'):
                search_result.append(
                    f'・検索結果{i}は以下の通りです。\n{doc.metadata["split_source"]}\n\n')
                i += 1
        return search_result

    serach_result = search_database(search_word)
    return serach_result


search_tools = [search]


class ClassAgentInput(BaseModel):
    user_utterance: str = Field(
        description="The user's most recent utterance that is communicated to the person in charge of the school database search.")


class ClassAgent(BaseToolAgent):
    def __init__(self, llm, memory, chat_history, verbose):
        super().__init__(llm, memory, chat_history, verbose)
        # ClassAgent 特有の初期化（もしあれば）

    def run(self, input):
        # ClassAgent特有の処理
        class_agent = self.initialize_agent(
            agent_type=AgentType.OPENAI_FUNCTIONS,
            tools=search_tools,  # 事前に定義されたsearch関数
            system_message_template=SEARCHDB_SYSTEM_PROMPT
        )
        return class_agent.run(input)
