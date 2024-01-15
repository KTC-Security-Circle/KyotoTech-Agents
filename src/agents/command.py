import os

from . import tools

class CheckCommandParam:

    def __init__(self, check_command_bool: bool = False, command: str = ""):
        self.check_command_bool = check_command_bool
        self.command = command


def check_command(user_message: str) -> CheckCommandParam:
    if user_message.startswith('/'):
        space_index = user_message.find(' ')
        if space_index != -1:
            command = user_message[1:space_index]
            param = CheckCommandParam(check_command_bool=True, command=command)
        else:
            param = CheckCommandParam()
    else:
        param = CheckCommandParam()
    return param

class Command:

    def __init__(self, llm, memory, chat_history, verbose):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose


    def run(self, command, user_message: str) -> str:
        if command == "test":
            return "test"
        elif command == "help":
            return_text = """コマンド一覧"""
            return return_text
        elif command == "search":
            pass
        elif command == "procedure":
            agent = tools.ProcedureAgent(llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
            return agent.run(user_message)




