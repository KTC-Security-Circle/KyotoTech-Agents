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

    def __init__(self, llm, memory, readonly_memory, chat_history, verbose):
        self.llm = llm
        self.memory = memory
        self.readonly_memory = readonly_memory
        self.chat_history = chat_history
        self.verbose = verbose


    def run(self, command: str, user_message: str) -> str:
        self.user_message = user_message.replace("/" + command, "", 1)
        
        if command == "help":
            return_text = """コマンド一覧
  ・/help : コマンド一覧を表示します。
  ・/search : 検索を行います。
  ・/procedure : 手続きを行います。
  ・/horoscope : 星占いを行います。

以上の中から選択し、”/コマンド名+半角スペース” で実行できます。
            """
            return return_text
        elif command == "search":
            agent = tools.SearchAgent(llm=self.llm, memory=self.memory, readonly_memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
            return agent.run(self.user_message)
        elif command == "procedure":
            agent = tools.ProcedureAgent(
                llm=self.llm, memory=self.memory, readonly_memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
            return agent.run(self.user_message)
        elif command == "horoscope":
            agent = tools.HoroscopeAgent(llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
            return agent.run(self.user_message)
        else:
            return_text = """{{ command }} というコマンドは見つかりませんでした。
コマンドは ”/コマンド名+半角スペース” で実行できます。
コマンド名がわからない場合は /help でコマンド一覧を確認できます。
"""
            return_text = return_text.format(command=command)
            return return_text