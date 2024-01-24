from langchain.agents import Tool
from pydantic.v1 import BaseModel, Field

from ...template.agent_model import BaseDispatcherAgent
from ...tools import procedure


class ProcedureAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")


DEFAULT_ANSWER = """私が行うことのできる各種申請は以下の通りです。
・公欠届
・遅延届

○○を申請したいと言ってもらえれば、詳細を聞き申請をすることができます。
"""

def default_answer(input):
    return DEFAULT_ANSWER

class ProcedureAgent(BaseDispatcherAgent):
    def __init__(self, llm, memory, readonly_memory, chat_history, verbose):
        super().__init__(llm, memory, readonly_memory, chat_history, verbose)

    def define_tools(self):
        self.default_answer = default_answer
        self.late_notification_agent = procedure.LateNotificationAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        self.official_absence_agent = procedure.OfficialAbsenceAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)

        procedure_agent_tools = [
            Tool.from_function(
                func=self.late_notification_agent.run,
                name="late_notification",
                # description="この担当者は遅延届申請を受け付けている担当者。ユーザーが遅延届を届け出たい場合はこの担当者に任せる。",
                description="This person is the person in charge of accepting late report applications. If a user wants to report a delay, this person is in charge.",
                args_schema=procedure.LateNotificationAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.official_absence_agent.run,
                name="official_absence",
                # description="この担当者は公欠届申請を受け付けている担当者。ユーザーが公欠届を届け出たい場合はこの担当者に任せる。"
                description="This person is the person in charge of accepting public absence notification applications. If the user wants to report a public absence, this person is in charge.",
                args_schema=procedure.OfficialAbsenceAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.default_answer,
                name="DEFAULT",
                # description="この担当者は専門的な会話ではない場合に任せる担当者。"
                description="This person is the person to leave in charge when the conversation is not a professional one.",
                return_direct=True
            ),
        ]
        return procedure_agent_tools
