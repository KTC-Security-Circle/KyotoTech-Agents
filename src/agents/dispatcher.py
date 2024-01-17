from langchain.agents import Tool

from .template.agent_model import BaseDispatcherAgent
from . import tools


class MainDispatcherAgent(BaseDispatcherAgent):
    def __init__(self, llm, memory, readonly_memory, chat_history, verbose):
        super().__init__(llm, memory, readonly_memory, chat_history, verbose)
    
    def define_tools(self):
        self.search_agent = tools.SearchAgent(
            llm=self.llm, memory=self.readonly_memory, readonly_memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        self.procedure_agent = tools.ProcedureAgent(
            llm=self.llm, memory=self.readonly_memory, readonly_memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        self.horoscope_agent = tools.HoroscopeAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        self.default_agent = tools.DefaultAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        
        main_dispatcher_tools = [
            Tool.from_function(
                func=self.search_agent.run,
                name="search",
                description="This person is in charge of database searches. This person should be responsible for responding to conversations related to searches, questions, and doubts.",
                args_schema=tools.SearchAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.procedure_agent.run,
                name="procedure",
                description="This person is in charge of various procedures. This person should be responsible for handling conversations related to various procedures.",
                args_schema=tools.ProcedureAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.horoscope_agent.run, # ラッパー関数を指定, ここで定義した関数が実行される
                name="horoscope", # ツールの名前を指定, この名前がディスパッチャーエージェントの出力になる, この名前が出力された際にfuncで指定した関数が実行される
                description="This is the person in charge of astrology. This person should be in charge of handling conversations related to horoscopes.", # ツールの説明を指定, この説明をもとにディスパッチャーエージェントはユーザーに対して適切なツールを選択する
                args_schema=tools.HoroscopeAgentInput, # ツールの入力の定義を指定, この定義をもとにディスパッチャーエージェントはユーザーからの入力をツールに渡す
                return_direct=True # ツールの出力を直接返すかどうかを指定, Trueの場合はツールの出力をそのまま返す, Falseの場合はツールの出力をディスパッチャーエージェントの入力として再度渡す
            ),
            Tool.from_function(
                func=self.default_agent.run,
                name="DEFAULT",
                description="This is the person in charge of general conversations. This person should be assigned to handle conversations that are general and should not be left to a specific expert.",
                args_schema=tools.DefaultAgentInput,
                return_direct=True
            ),
        ]
        
        return main_dispatcher_tools
