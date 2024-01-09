import langchain
from langchain.agents import AgentType, initialize_agent, tool
from langchain.prompts.chat import SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field

from ..template import default_value


# プロンプトの設定
# DEFAULT_SYSTEM_PROMPT = '''あなたは会話型アシスタントエージェントです。
# 次に与えるあなたの role になりきってユーザーと会話してください。

# # role
# - あなたはアシスタントエージェントの "KyotoTECH君" です。
# - あなたが働いている会社は "京都デザイン＆テクノロジー専門学校" で、よく "京都テック" と訳されます。
# - あなたの仕事はユーザーとあなたとの会話内容を読み、フレンドリーな会話を行うことです。
# - ユーザーにあなたができることを尋ねられた場合は、ask_can_do という関数を使ってユーザーにあなたができることを教えてください。
# '''
DEFAULT_SYSTEM_PROMPT = '''You are a conversational assistant agent.
Please embody the role provided next and engage in a conversation with the user.
Respond in Japanese.

# role
- You are "KyotoTECH君", an assistant agent.
- You work for "京都デザイン＆テクノロジー専門学校", often translated as "京都テック".
- Your job is to read the conversation content between you and the user and engage in friendly dialogue.
- If the user asks what you are capable of doing, use the function named 'ask_can_do' to inform the user about your abilities.
'''


@tool("ask_can_do", return_direct=True)  # Agentsツールを作成。
def ask_can_do():  # ユーザーにあなたができることを教える関数を作成。
    """I will tell you what you can do. If you are asked what you can do, perform this function."""
    return_messages = '''私は様々なことができますが、例えば以下のようなことができますよ。
・学校の情報や奨学金についての情報にアクセスして、学校の情報を教えます。
・授業についての疑問や質問に答えます。
・公欠届や遅延届の作成から提出までを行います。
・現在の図書質の貸出状況を確認したり、おススメの本を紹介します。

是非、私に色々なことを聞いてみてくださいね！
'''
    return return_messages


default_tools = [ask_can_do]

class DefaultAgentInput(BaseModel): # デフォルトエージェントの入力の定義
    user_utterance: str = Field(
        description="This is the user's most recent utterance that communicates general content to the person in charge.")


class DefaultAgent:

    def __init__(
        self,
        llm: AzureChatOpenAI = default_value.default_llm,
        memory: ConversationBufferMemory = default_value.default_memory,
        chat_history: MessagesPlaceholder = default_value.default_chat_history,
        verbose: bool = False,
        ):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose
        
        # デバッグモードの設定
        langchain.debug = self.verbose
        
    def run(self, input):
        self.agent_kwargs = {
            "system_message": SystemMessagePromptTemplate.from_template(template=DEFAULT_SYSTEM_PROMPT),
            "extra_prompt_messages": [self.chat_history]
        }
        self.default_agent = initialize_agent(
            tools=default_tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=self.verbose,
            agent_kwargs=self.agent_kwargs,
            memory=self.memory
        )
        return self.default_agent.run(input)


# debag
# print(Agent.run("あなたについて教えて"))
