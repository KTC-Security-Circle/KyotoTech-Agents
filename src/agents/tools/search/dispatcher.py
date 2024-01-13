import langchain
from langchain_openai import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.agents import BaseSingleActionAgent,  Tool,  AgentExecutor
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
from langchain.memory import ReadOnlySharedMemory

from ...template import default_value, agent_model
from ...tools import search




class SearchAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")


class SearchAgent:

    def __init__(
        self,
        llm: AzureChatOpenAI = default_value.default_llm,
        memory: ConversationBufferMemory = default_value.default_memory,
        readonly_memory: ReadOnlySharedMemory = default_value.default_readonly_memory,
        chat_history: MessagesPlaceholder = default_value.default_chat_history,
        verbose: bool = False,
    ):
        self.llm = llm
        self.memory = memory
        self.readonly_memory = readonly_memory
        self.chat_history = chat_history
        self.verbose = verbose

        langchain.debug = self.verbose

        self.defalt_answer = (
            f'私が行うことのできる各種検索は以下の通りです。\n'
            f'・学校データ\n'
            f'・奨学金データ\n'
            f'・授業データ\n'
            f'○○を申請したいと言ってもらえれば、詳細を聞き申請をすることができます。'
        )

        def defalt_answer_wrapper(input):
            return self.defalt_answer()
        
        self.school_agent = search.SchoolAgent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def school_agent_wrapper(input):
            return self.late_notification_agent.run(input)
        
        self.class_agent = search.ClassAgent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def class_agent_wrapper(input):
            return self.official_absence_agent.run(input)
        
        self.scholarship_agent = search.ScholarshipAgent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def scholarship_agent_wrapper(input):
            return self.official_absence_agent.run(input)

        self.tools = [  # ツールのリスト
            Tool.from_function(
                func=school_agent_wrapper,
                name="school_agent",
                # description="学校の概要について検索をする担当者です。学校の概要の検索に関係する会話の対応はこの担当者にまかせるべきです。", # 日本語ver
                description="",  # 英語ver
                args_schema=search.SchoolAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=class_agent_wrapper,
                name="class_agent",
                # description="公欠届の申請に関する担当者です。公欠届に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is the person in charge regarding the application for the Notification of Public Absence. This person should be the person in charge of handling conversations related to the public absence notification.",  # 英語ver
                args_schema=search.ClassAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=scholarship_agent_wrapper,
                name="scholarship_agent",
                # description="公欠届の申請に関する担当者です。公欠届に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is the person in charge regarding the application for the Notification of Public Absence. This person should be the person in charge of handling conversations related to the public absence notification.",  # 英語ver
                args_schema=search.ScholarshipAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=defalt_answer_wrapper,
                name="DEFAULT",
                # description="特定の申請に関する情報がない場合はこの担当者に任せるべきです。", # 日本語ver
                description="If you do not have information on a particular application, this person should be assigned to you.",  # 英語ver
                return_direct=True
            ),
        ]

    def run(self, input: str):
        self.dispatcher_agent = agent_model.DispatcherAgent(  # ディスパッチャーエージェントの初期化
            chat_model=self.llm, readonly_memory=self.readonly_memory, tools=self.tools, verbose=self.verbose)
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=self.dispatcher_agent, tools=self.tools, memory=self.memory, verbose=self.verbose
        )
        return self.agent.run(input)


# while(True):
#     message = input(">> ")
#     if message == "exit" or message == ":q":
#         break
#     try:
#         output = agent.run(message)
#         print(output)
#     except Exception as e:
#         print(e)
