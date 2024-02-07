from operator import itemgetter

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.agents import AgentType, tool
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
import json
import requests
import datetime
from pydantic.v1 import BaseModel, Field

from tech_agents.template.agent_model import BaseToolAgent

# プロンプトの設定
# DEFAULT_SYSTEM_PROMPT = '''あなたは会話型アシスタントエージェントです。
# 次に与えるあなたの role になりきってユーザーと会話してください。

# # role
# - あなたは翻訳家です。
# - あなたの仕事はユーザーから送られてきた文章または単語を翻訳し的確に表現することです。
# - 与えられた文章を、日本語から英語、英語から日本語、または必要に応じて他の言語にも翻訳してください。
# - 文中の略語や頭字語はそのままの形で残すようお願いします。
# - 不明点や翻訳中に判断が難しい部分があれば、可能な限り詳しく質問してください。
# '''
DEFAULT_SYSTEM_PROMPT = '''You are a conversational assistant agent.
Please embody the role provided next and engage in a conversation with the user.

# role
- you are a translator.
- Your job is to translate sentences or words sent by users and express them as they are.
- Translate the given sentences from Japanese to English, English to Japanese, or other languages as needed.
- Please leave all abbreviations and acronyms in the text as they are.
- If there are any unclear points or parts that are difficult to judge during translation, please ask in as much detail as possible.
'''

translate_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate.from_template(template=DEFAULT_SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)

class TranslateAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the translator.")


class TranslateAgent:
    def __init__(self, llm, memory, chat_history, verbose):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose

    def run(self, input):
        history = self.memory.load_memory_variables({})['chat_history']
        transrate_chain = translate_prompt | self.llm
        inputs = {"chat_history": history, "input": input}
        result = transrate_chain.invoke(inputs)
        return result.content

