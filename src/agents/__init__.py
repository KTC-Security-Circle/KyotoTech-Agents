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

from agents.dispatcher import MainAgent


# デバッグモードを有効
verbose = True
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
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
readonly_memory = ReadOnlySharedMemory(memory=memory)
chat_history = MessagesPlaceholder(variable_name='chat_history')



def run(user_message: str):
    try:
        main_agent = MainAgent(
            llm=default_llm,
            memory=readonly_memory,
            chat_history=chat_history,
            verbose=verbose
        )
        output = main_agent.run(user_message)
        return output
    except Exception as e:
        print("err : " + str(e))
        err_msg = f"エラーが発生しました。時間をおいて再度お試しください。"
        return err_msg

def test():
    message = "こんにちは"
    main_agent = MainAgent(
        llm=default_llm,
        memory=readonly_memory,
        chat_history=chat_history,
        verbose=verbose
    )
    print(main_agent.run(message))