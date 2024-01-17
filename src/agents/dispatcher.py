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
                # description="この担当者は検索や質問に答える担当者。技術的な質問や、学校についての質問、奨学金についての疑問などを解決する場合はこの担当者に任せる。",
                description="This person is in charge of searching and answering questions. If you have a technical question, a question about the school, or a question about scholarships, this is the person to contact.",
                args_schema=tools.SearchAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.procedure_agent.run,
                name="procedure",
                # description="この担当者は各種手続きに関する担当者。公欠届や遅延届などの手続きに関する会話の対応はこの担当者に任せる。",
                description="This person is in charge of various procedures. This person is in charge of handling conversations regarding procedures such as public absence reports and late reports.",
                args_schema=tools.ProcedureAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=self.horoscope_agent.run, # ラッパー関数を指定, ここで定義した関数が実行される
                name="horoscope", # ツールの名前を指定, この名前がディスパッチャーエージェントの出力になる, この名前が出力された際にfuncで指定した関数が実行される
                # description="この担当者は星占いのできる担当者。星占いがしたい時はこの担当者に任せる。", # ツールの説明を指定, この説明をもとにディスパッチャーエージェントはユーザーに対して適切なツールを選択する
                description="This person in charge is the person in charge who can do horoscopes. When you want to do horoscopes, leave it to this person in charge.", # ツールの説明を指定, この説明をもとにディスパッチャーエージェントはユーザーに対して適切なツールを選択する
                args_schema=tools.HoroscopeAgentInput, # ツールの入力の定義を指定, この定義をもとにディスパッチャーエージェントはユーザーからの入力をツールに渡す
                return_direct=True # ツールの出力を直接返すかどうかを指定, Trueの場合はツールの出力をそのまま返す, Falseの場合はツールの出力をディスパッチャーエージェントの入力として再度渡す
            ),
            Tool.from_function(
                func=self.default_agent.run,
                name="DEFAULT",
                # description="この担当者は専門的な会話ではない場合に任せる担当者。",
                description="This person is the person to leave in charge when the conversation is not a professional one.",
                args_schema=tools.DefaultAgentInput,
                return_direct=True
            ),
        ]
        
        return main_dispatcher_tools
