

from langchain.agents import Tool
from pydantic.v1 import BaseModel, Field

from ...template.agent_model import BaseDispatcherAgent
from ...tools import search


class SearchAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")

def default_answer(input):
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
                # description="この担当者は京都テックという名前の学校の受付担当者。京都テックについて聞かれた場合や専攻等について聞かれた場合はこの担当者に任せる。", # 日本語ver
                description="This person is the receptionist for the school named Kyoto Tech. If you are asked about Kyoto Tech, your major, etc., leave it to this person.",  # 英語ver
                args_schema=search.SchoolAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.class_agent.run,
                name="class_agent",
                # description="この担当者は学校の先生をまとめる担当者。授業のことについてや技術的な質問について聞かれた場合はこの担当者に任せる。", # 日本語ver
                description="This person is in charge of organizing the teachers at the school. If you are asked about a class or about technical questions, this person is in charge.",  # 英語ver
                args_schema=search.ClassAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.scholarship_agent.run,
                name="scholarship_agent",
                # description="この担当者は奨学金についての相談受付担当者。奨学金について聞かれた場合はこの担当者に任せる。", # 日本語ver
                description="This person is the person in charge of counseling about the scholarship. If you are asked about scholarships, this person will be your contact person.",  # 英語ver
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
