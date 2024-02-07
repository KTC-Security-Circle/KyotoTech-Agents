import langchain
from langchain.agents import AgentType, initialize_agent, tool
from langchain.prompts.chat import SystemMessagePromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.schema import HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from pydantic.v1 import BaseModel, Field


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


class DefaultInput: # デフォルトエージェントの入力の定義
    user_utterance: str = Field(
        description="This is the user's most recent utterance that communicates general content to the person in charge.")


class TranslateAgent:
    def __init__(self, llm=None, memory=None, chat_history=None, verbose=False):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose

    def translate(self, input_text):
        # TranselateAgent特有の処理
        translate_agent = self.initialize_agent(
            agent_type=AgentType.OPENAI_FUNCTIONS,
            system_message_template=DEFAULT_SYSTEM_PROMPT
        )
        return "Translated text: " + input_text

agent = TranslateAgent()

text_to_translate = input("Translate text: ")

translated_text = agent.translate(text_to_translate)
print(translated_text)

# debag