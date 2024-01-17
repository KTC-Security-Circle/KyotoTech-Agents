from langchain.agents import Tool
from pydantic.v1 import BaseModel, Field

from ...template.agent_model import BaseDispatcherAgent
from ...tools import procedure


class ProcedureAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")

def default_answer():
    message = (
            f'私が行うことのできる各種申請は以下の通りです。\n'
            f'・公欠届\n'
            f'・遅延届\n'
            f'○○を申請したいと言ってもらえれば、詳細を聞き申請をすることができます。'
        )
    return message

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
                description="This person is in charge of late notifications. This person should be responsible for handling conversations related to late notifications.",
                args_schema=procedure.LateNotificationAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.official_absence_agent.run,
                name="official_absence",
                description="This person is in charge of official absences. This person should be responsible for handling conversations related to official absences.",
                args_schema=procedure.OfficialAbsenceAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.default_answer,
                name="DEFAULT",
                description="This is the person in charge of general conversations. This person should be assigned to handle conversations that are general and should not be left to a specific expert.",
                return_direct=True
            ),
        ]
        return procedure_agent_tools
