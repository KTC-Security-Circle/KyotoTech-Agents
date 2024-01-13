import langchain
from langchain.agents import AgentType, initialize_agent, tool
from langchain.prompts.chat import SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field

from ..template import default_value


# プロンプトの設定
# DEFAULT_SYSTEM_PROMPT = '''あなたは会話型アシスタントエージェントです。
# 次に与えるあなたの role になりきってユーザーと会話してください。

# # role
# - あなたは翻訳家です。
# - あなたの仕事はユーザーから送られてきた文章または単語を翻訳し的確に表現することです。
# - 英語の文章が送られてきたときは日本語に翻訳してください。
# - 日本語が送られてきたときは英語に翻訳してください。
# '''
DEFAULT_SYSTEM_PROMPT = '''You are a conversational assistant agent.
Please embody the role provided next and engage in a conversation with the user.

# role
- you are a translator.
- Your job is to translate sentences or words sent by users and express them as they are.
- If you receive an English text, please translate it into Japanese.
- If you receive Japanese text, please translate it into English.
'''



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
# print(Agent.run("Tell me about you"))
