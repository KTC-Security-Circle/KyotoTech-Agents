from langchain.chat_models import AzureChatOpenAI
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
from pydantic.v1 import Extra
from typing import Any, List, Tuple, Set, Union

from .template import default_value
from . import tools





# プロンプトテンプレートの定義
# 日本語ver
# ROUTER_TEMPLATE = '''あなたの仕事はユーザーとあなたとの会話内容を読み、
# 以下の選択候補からその説明を参考にしてユーザーの対応を任せるのに最も適した候補を選び、その名前を回答することです。
# あなたが直接ユーザーへ回答してはいけません。あなたは対応を任せる候補を選ぶだけです。

# << 選択候補 >>
# 名前: 説明
# {destinations}

# << 出力形式の指定 >>
# 選択した候補の名前のみを出力して下さい。
# 注意事項: 出力するのは必ず選択候補として示された候補の名前の一つでなければなりません。
# ただし全ての選択候補が不適切であると判断した場合には "DEFAULT" とすることができます。

# << 回答例 >>
# 「あなたについて教えて下さい。」と言われても返事をしてはいけません。
# 選択候補に適切な候補がないケースですから"DEFAULT"と答えて下さい。
# '''

# 英語ver
ROUTER_TEMPLATE = '''Your job is to read the conversation between the user and yourself, and based on the descriptions provided below, select the most suitable candidate to handle the user's response.
You should not directly answer the user; your role is solely to choose the appropriate candidate.

# Choices
Name: Description
{destinations}

# Output Format
Please output only the name of the selected candidate.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT".

# Example Answer
If asked, 'Tell me about yourself,' you should not respond.
Since there is no appropriate candidate in the choices, answer with "DEFAULT".
'''

ROUTER_PROMPT_SUFFIX = '''<< Output Format Specification >>
I'll reiterate the instructions one last time. Please output only the name of the candidate you have selected.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT".
'''



# BaseOutputParserをもとにしたDestinationOutputParserクラスの定義
# このクラスでは、ディスパッチャーエージェントの出力を解析し、適切なツールを選択します。
# もし、ディスパッチャーエージェントの出力が適切なツールを選択できていない場合は、エラーを返します。
class DestinationOutputParser(BaseOutputParser[str]):  
    destinations: Set[str] # 解析対象のツールの名前を指定します。

    class Config:  # 内部クラスConfigを定義します。このクラスはPydanticモデルの設定を制御します。
        extra = Extra.allow # extraをallowに設定することで、このクラスに未定義のフィールドが存在してもエラーを返さないようにします。

    def __init__(self, **kwargs): # コンストラクタを定義します。
        super().__init__(**kwargs) # 親クラスのコンストラクタを呼び出します。
        self.destinations_and_default = list(self.destinations) + ["DEFAULT"] # 解析対象のツールの名前とデフォルトの名前をリストに格納します。

    def parse(self, text: str) -> str: # 与えられたtextを解析し、適切なツールを選択します。
        matched = [int(d in text) for d in self.destinations_and_default] # 解析対象のツールの名前とデフォルトの名前がtextに含まれているかどうかを判定します。(含まれていれば1, 含まれていなければ0)
        if sum(matched) != 1:  # textがdestinations_and_defaultの要素と一致しないか、2つ以上の要素と一致する場合はエラーを返します。
            raise OutputParserException(
                f"DestinationOutputParser expected output value includes "
                f"one(and only one) of {self.destinations_and_default}. "
                f"Received {text}."
            )

        # matchedの中で1のインデックスを取得し、そのインデックスに対応するdestinations_and_defaultの要素を返します。
        return self.destinations_and_default[matched.index(1)]

    @property
    def _type(self) -> str:
        return "destination_output_parser"


# ディスパッチャーエージェントの定義
# このエージェントは、複数のエージェントを統括し、ユーザーからの入力に対して適切なエージェントを選択して実行します。
class DispatcherAgent(BaseSingleActionAgent):

    chat_model: BaseChatModel
    readonly_memory: ReadOnlySharedMemory
    tools: List[Tool]
    verbose: bool = False

    class Config:
        extra = Extra.allow

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        destinations = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        router_template = ROUTER_TEMPLATE.format(destinations=destinations)
        router_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                template=router_template),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate(prompt=PromptTemplate(
                input_variables=['input'], template='{input}')),
            SystemMessagePromptTemplate.from_template(
                template=ROUTER_PROMPT_SUFFIX)
        ])
        self.router_chain = LLMChain(
            llm=self.chat_model,
            prompt=router_prompt_template,
            memory=self.readonly_memory,
            verbose=self.verbose
        )

        self.route_parser = DestinationOutputParser(
            destinations=set([tool.name for tool in self.tools])
        )

    @property
    def input_keys(self):
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:

        router_output = self.router_chain.run(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException as ope:
            destination = "DEFAULT"

        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:

        router_output = await self.router_chain.arun(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException as ope:
            destination = "DEFAULT"

        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")




class Agent:
    '''
    メインのディスパッチャーエージェントの実行クラスです。
    このエージェントは、複数のエージェントを統括し、ユーザーからの入力に対して適切なエージェントを選択して実行します。
    '''

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
        
        
        # デフォルトエージェントの定義
        self.default_agent = tools.DefaultAgent( # デフォルトエージェントの初期化、ここでデフォルトエージェントの引数を設定する
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        def default_agent_wrapper(user_message): # デフォルトエージェントのラッパー関数の定義 これを使って定義したデフォルトエージェントを実行する
            ai_message = self.default_agent.run(user_message) # デフォルトエージェントを実行し、その出力を変数に格納する
            return ai_message
        
        
        # 星占いエージェントの定義
        self.horoscope_agent = tools.HoroscopeAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        def horoscope_agent_wrapper(user_message):
            ai_message = self.horoscope_agent.run(user_message)
            return ai_message

        
        # データベース検索エージェントの定義
        self.search_database_agent = tools.SearchDBAgent(
            llm=self.llm, memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        def search_database_agent_wrapper(user_message):
            ai_message = self.search_database_agent.run(user_message)
            return ai_message

        
        # 各種申請エージェントの定義
        self.procedure_agent = tools.ProcedureAgent(
            llm=self.llm, memory=self.readonly_memory, readonly_memory=self.readonly_memory, chat_history=self.chat_history, verbose=self.verbose)
        def procedure_agent_wrapper(user_message):
            ai_message = self.procedure_agent.run(user_message)
            return ai_message



        # 使用ツールの定義
        self.tools = [
            Tool.from_function(
                func=horoscope_agent_wrapper, # ラッパー関数を指定, ここで定義した関数が実行される
                name="horoscope", # ツールの名前を指定, この名前がディスパッチャーエージェントの出力になる, この名前が出力された際にfuncで指定した関数が実行される
                description="This is the person in charge of astrology. This person should be in charge of handling conversations related to horoscopes.", # ツールの説明を指定, この説明をもとにディスパッチャーエージェントはユーザーに対して適切なツールを選択する
                args_schema=tools.HoroscopeAgentInput, # ツールの入力の定義を指定, この定義をもとにディスパッチャーエージェントはユーザーからの入力をツールに渡す
                return_direct=True # ツールの出力を直接返すかどうかを指定, Trueの場合はツールの出力をそのまま返す, Falseの場合はツールの出力をディスパッチャーエージェントの入力として再度渡す
            ),
            Tool.from_function(
                func=search_database_agent_wrapper,
                name="search_database",
                description="This person is in charge of school database searches. This person should be responsible for searching the school database and handling conversations related to school information.",
                args_schema=tools.SearchDBAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=procedure_agent_wrapper,
                name="procedure",
                description="This person is in charge of various procedures. This person should be responsible for handling conversations related to various procedures.",
                args_schema=tools.ProcedureAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=default_agent_wrapper,
                name="DEFAULT",
                description="This is the person in charge of general conversations. This person should be assigned to handle conversations that are general and should not be left to a specific expert.",
                args_schema=tools.DefaultAgentInput,
                return_direct=True
            ),
        ]
    
    
    def run(self, user_message: str) -> str:
        '''
        定義したディスパッチャーエージェントを実行するメソッドです。
        このメソッドは、ディスパッチャーエージェントを実行し、その出力を返します。
        '''
        # ディスパッチャーエージェントの初期化
        dispatcher_agent = DispatcherAgent(
            chat_model=self.llm, readonly_memory=self.readonly_memory, tools=self.tools, verbose=self.verbose)
        agent = AgentExecutor.from_agent_and_tools(
            agent=dispatcher_agent, tools=self.tools, memory=self.memory, verbose=self.verbose
        )
        return agent.run(user_message)

