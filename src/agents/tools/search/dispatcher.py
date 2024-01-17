

from langchain.agents import Tool
from pydantic.v1 import BaseModel, Field

from ...template.agent_model import BaseDispatcherAgent
from ...tools import search


class SearchAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")

def default_answer():
    message = (
            f'私が行うことのできる各種検索は以下の通りです。\n'
            f'・学校データ\n'
            f'・奨学金データ\n'
            f'・授業データ\n'
            f'○○について検索したいと言ってもらえれば、詳細を聞きお答えすることができます。'
        )
    return message


class SearchAgent(BaseDispatcherAgent):
    def __init__(self, llm, memory, readonly_memory, chat_history, verbose):
        super().__init__(llm, memory, readonly_memory, chat_history, verbose)

    def define_tools(self):
        self.default_answer = default_answer
        self.school_agent = search.SchoolAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        self.class_agent = search.ClassAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        self.scholarship_agent = search.ScholarshipAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)

        search_agent_tools = [
            Tool.from_function(
                func=self.school_agent.run,
                name="school_agent",
                # description="学校の概要について検索をする担当者です。学校や京都テックについての検索に関係する会話の対応はこの担当者にまかせるべきです。", # 日本語ver
                description="This person is in charge of searching for information about the school. This person should be entrusted to handle conversations related to your search about the school and Kyoto Tech.",  # 英語ver
                args_schema=search.SchoolAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.class_agent.run,
                name="class_agent",
                # description="授業のことに関する担当者です。授業についての質問や授業データの検索に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is the person in charge regarding class matters. This person should be the person to contact for questions about the class and for handling conversations related to the retrieval of class data.",  # 英語ver
                args_schema=search.ClassAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.scholarship_agent.run,
                name="scholarship_agent",
                # description="奨学金のことに関する担当者です。奨学金についての質問や奨学金データの検索に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is your contact person regarding scholarship matters. This person should be the one to handle any questions about the scholarship and any conversations related to the scholarship data search.",  # 英語ver
                args_schema=search.ScholarshipAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.default_answer,
                name="DEFAULT",
                # description="特定の検索に関する情報がない場合はこの担当者に任せるべきです。", # 日本語ver
                description="If you do not have information on a particular application, this person should be assigned to you.",  # 英語ver
                return_direct=True
            ),
        ]
        return search_agent_tools
