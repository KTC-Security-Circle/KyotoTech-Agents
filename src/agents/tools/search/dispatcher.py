import langchain
from langchain_openai import AzureChatOpenAI
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.agents import  Tool,  AgentExecutor
from langchain.prompts.chat import MessagesPlaceholder
from pydantic.v1 import BaseModel, Field
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
            f'○○について検索したいと言ってもらえれば、詳細を聞きお答えすることができます。'
        )

        def defalt_answer_wrapper(input):
            return self.defalt_answer
        
        self.school_agent = search.SchoolAgent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def school_agent_wrapper(input):
            return self.school_agent.run(input)
        
        self.class_agent = search.ClassAgent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def class_agent_wrapper(input):
            return self.class_agent.run(input)
        
        self.scholarship_agent = search.ScholarshipAgent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def scholarship_agent_wrapper(input):
            return self.scholarship_agent.run(input)

        self.tools = [  # ツールのリスト
            Tool.from_function(
                func=school_agent_wrapper,
                name="school_agent",
                # description="学校の概要について検索をする担当者です。学校や京都テックについての検索に関係する会話の対応はこの担当者にまかせるべきです。", # 日本語ver
                description="This person is in charge of searching for information about the school. This person should be entrusted to handle conversations related to your search about the school and Kyoto Tech.",  # 英語ver
                args_schema=search.SchoolAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=class_agent_wrapper,
                name="class_agent",
                # description="授業のことに関する担当者です。授業についての質問や授業データの検索に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is the person in charge regarding class matters. This person should be the person to contact for questions about the class and for handling conversations related to the retrieval of class data.",  # 英語ver
                args_schema=search.ClassAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=scholarship_agent_wrapper,
                name="scholarship_agent",
                # description="奨学金のことに関する担当者です。奨学金についての質問や奨学金データの検索に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is your contact person regarding scholarship matters. This person should be the one to handle any questions about the scholarship and any conversations related to the scholarship data search.",  # 英語ver
                args_schema=search.ScholarshipAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=defalt_answer_wrapper,
                name="DEFAULT",
                # description="特定の検索に関する情報がない場合はこの担当者に任せるべきです。", # 日本語ver
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
