import os

import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder

from agents.dispatcher import Agent


# デフォルトのLLMの定義
default_llm = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
    temperature=0,
    model_kwargs={"top_p": 0.1}
)

# 会話メモリの定義
default_memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
default_chat_history = MessagesPlaceholder(variable_name='chat_history')


class MainAgent:
    llm: AzureChatOpenAI
    memory: ConversationBufferMemory
    chat_history: MessagesPlaceholder
    verbose: bool

    """
    MainAgentクラスは、メインのエージェントを表すクラスです。
    このクラスは、AzureChatOpenAI、ConversationBufferMemory、MessagesPlaceholderなどの属性を持ちます。
    メインエージェントは、指定された入力に対してAgentクラスを実行します。
    """

    def __init__(
        self,
        llm: AzureChatOpenAI = default_llm,
        memory: ConversationBufferMemory = default_memory,
        chat_history: MessagesPlaceholder = default_chat_history,
        verbose: bool = False,
    ):
        """
        MainAgentクラスのコンストラクタです。
        デフォルトの引数を使用して、AzureChatOpenAI、ConversationBufferMemory、MessagesPlaceholder、verboseを初期化します。
        
        インスタンス化
        ------------
        main_agent = MainAgent(
            llm=あなたの使用したいLLM,
            memory=あなたの使用したいメモリ,
            chat_history=あなたの使用したい会話履歴,
            verbose=デバッグモードを有効にするかどうか
        )
        
        実行
        ------------
        message = "こんにちは"
        output = main_agent.run(message)
        print(output)
        
        """

        # 引数の初期化
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose

        # メモリの読み取り専用化
        self.readonly_memory = ReadOnlySharedMemory(memory=self.memory)
        # デバッグモードの設定
        langchain.debug = self.verbose

    def run(self, user_message: str) -> str:
        """
        メインエージェントを実行するメソッドです。
        Agentクラスを生成し、指定された入力を渡して実行します。
        """
        main_agent = Agent(
            llm=self.llm,
            memory=self.memory,
            readonly_memory=self.readonly_memory,
            chat_history=self.chat_history,
            verbose=self.verbose
        )
        return main_agent.run(user_message)
