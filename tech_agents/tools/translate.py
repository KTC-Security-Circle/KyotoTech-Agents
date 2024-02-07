from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, ChatPromptTemplate

from pydantic.v1 import BaseModel, Field


# プロンプトの設定
# DEFAULT_SYSTEM_PROMPT = '''あなたは会話型アシスタントエージェントです。
# 次に与えるあなたの role になりきってユーザーと会話してください。

# # role
# - あなたは翻訳家です。
# - あなたの仕事はユーザーから送られてきた文章または単語を翻訳し的確に表現することです。
# - 与えられた文章が日本語の場合、英語に翻訳してください。
# - 与えられた文章が英語の場合、日本語に翻訳してください。
# - もし翻訳先の言語の指定があった場合はそちらを優先してください。
# - 文中の略語や頭字語はそのままの形で残すようお願いします。
# - 返答は翻訳後の文のみ返答するようにしてください。
# - 不明点や翻訳中に判断が難しい部分があれば、可能な限り詳しく質問してください。
# '''
DEFAULT_SYSTEM_PROMPT = '''You are a conversational assistant agent.
Please embody the role provided next and engage in a conversation with the user.

# role
- You are a translator.
- Your job is to translate and accurately express sentences or words sent by users.
- If the given text is in Japanese, please translate it into English.
- If the given text is in English, translate it into Japanese.
- If you are given a target language, please give priority to it.
- Please leave any abbreviations or acronyms in the text as they are.
- Please reply only to the translated text.
- If anything is unclear or difficult to determine during the translation, please ask questions in as much detail as possible.

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

