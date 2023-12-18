import os

import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder

# グローバル変数の定義


class grobal_value:
    verbose = True
    langchain.debug = verbose

    llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
        openai_api_base=os.environ["OPENAI_API_BASE"],
        openai_api_version=os.environ["OPENAI_API_VERSION"],
        deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
        openai_api_key=os.environ["OPENAI_API_KEY"],
        openai_api_type="azure",
        temperature=0,
        model_kwargs={"top_p": 0.1}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    readonly_memory = ReadOnlySharedMemory(memory=memory)
    chat_history = MessagesPlaceholder(variable_name='chat_history')
