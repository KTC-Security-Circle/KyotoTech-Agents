import os
from dotenv import load_dotenv
load_dotenv()

import langchain
from langchain.chains.llm import LLMChain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.agents import AgentType, initialize_agent, tool, AgentExecutor
from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseOutputParser,
    OutputParserException
)
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from pydantic.v1 import Extra, BaseModel, Field
from typing import Any, List, Tuple, Set, Union

from agents import dispatcher
from agents.tools import (
    horoscope,
    searchDB,
    default
)


class MainAgent:
    '''メインエージェントのクラスです。'''
    # デバッグモードを有効
    verbose = True 
    langchain.debug = verbose

    # Azure OpenAIのAPIを読み込み。
    llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
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


    # デフォルトエージェントの設定
    default_agent_kwargs = {
        "system_message": SystemMessagePromptTemplate.from_template(template=default.DEFAULT_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history]}
    default_agent = initialize_agent(
        default.default_tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=default_agent_kwargs,
        memory=readonly_memory
    )
    
    # 星占いエージェントの設定
    horoscope_agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(template=horoscope.HOROSCOPE_SYSTEM_PROMPT),
    "extra_prompt_messages": [chat_history]}
    horoscope_agent = initialize_agent(
        horoscope.horoscope_tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=horoscope_agent_kwargs,
        memory=readonly_memory
    )
    
    # 検索エージェントの設定
    search_agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(template=searchDB.SEARCHDB_SYSTEM_PROMPT),
    "extra_prompt_messages": [chat_history]}
    search_database_agent = initialize_agent(
        searchDB.search_tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=search_agent_kwargs,
        memory=readonly_memory
    )
    
    # メインエージェントの設定
    tools = [
        dispatcher.Tool.from_function(
            func=horoscope_agent.run,
            name="horoscope",
            description="This is the person in charge of astrology. This person should be in charge of handling conversations related to horoscopes.",
            args_schema=dispatcher.HoroscopeAgentInput,
            return_direct=True
        ),
        dispatcher.Tool.from_function(
            func=search_database_agent.run,
            name="searchDB",
            description="This person is in charge of school database searches. This person should be responsible for searching the school database and handling conversations related to school information.",
            args_schema=dispatcher.SearchDBAgentInput,
            return_direct=True
        ),
        dispatcher.Tool.from_function(
            func=default_agent.run,
            name="DEFAULT",
            description="This is the person in charge of general conversations. This person should be assigned to handle conversations that are general and should not be left to a specific expert.",
            args_schema=dispatcher.DefaultAgentInput,
            return_direct=True
        ),
    ]
    dispatcher_agent = dispatcher.DispatcherAgent(
        chat_model=llm, readonly_memory=readonly_memory, tools=tools, verbose=verbose)
    agent = AgentExecutor.from_agent_and_tools(
        agent=dispatcher_agent, tools=tools, memory=memory, verbose=verbose
    )

    def run(input: str):
        try:
            output = MainAgent.agent.run(input)
            return output
        except Exception as e:
            print("err : " + str(e))
            err_msg = f"エラーが発生しました。時間をおいて再度お試しください。"
            return err_msg
