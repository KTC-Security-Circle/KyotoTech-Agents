'''**agentsパッケージ**

このパッケージには、いくつかのエージェントをまとめたマルチエージェントの実装が含まれています。

# 使用方法
- 基本は `agents.run` 関数を使ってエージェントを実行します。
- その際、引数にはユーザーからの入力を与えてください。

- 個々のエージェントの実装は `agents.tools` サブパッケージにあります。

'''
import os

import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder

from agents.dispatcher import Agent


# デバッグモードを有効
verbose = False
langchain.debug = verbose

# Azure OpenAIのAPIを読み込み。
default_llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
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
    
    def __init__(
        self, 
        llm: AzureChatOpenAI = default_llm,
        memory: ConversationBufferMemory = default_memory,
        chat_history: MessagesPlaceholder = default_chat_history,
        verbose: bool = False,
        ):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose
        
        self.readonly_memory = ReadOnlySharedMemory(memory=self.memory)
        langchain.debug = self.verbose
        
        
        
    def run(self, input):
        main_agent = Agent(
            llm=self.llm,
            memory=self.memory,
            readonly_memory=self.readonly_memory,
            chat_history=self.chat_history,
            verbose=self.verbose
        )
        return main_agent.run(input)

