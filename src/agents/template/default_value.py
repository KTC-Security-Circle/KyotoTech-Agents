import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain_openai import AzureOpenAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from azure.search.documents.indexes.models import (
    SearchableField,
    SearchField,
    SearchFieldDataType,
    SimpleField,
)



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
    chunk_size=1
)


vector_store_address: str = os.environ["AZURE_SEARCH_ENDPOINT"]
vector_store_password: str = os.environ["AZURE_SEARCH_KEY"]
embeddings: AzureOpenAIEmbeddings = default_embeddings_model
embedding_function = embeddings.embed_query

