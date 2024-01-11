import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain_openai import AzureOpenAIEmbeddings



# デフォルトのLLMの定義
default_llm = AzureChatOpenAI(
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_type="azure",
    temperature=0,
    model_kwargs={"top_p": 0.1}
)

# デフォルトの会話メモリの定義
default_memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
default_chat_history = MessagesPlaceholder(variable_name='chat_history')
default_readonly_memory = ReadOnlySharedMemory(memory=default_memory)

default_embeddings_model = AzureOpenAIEmbeddings(
    azure_deployment=os.environ["DEPLOYMENT_EMBEDDINGS_NAME"],
)