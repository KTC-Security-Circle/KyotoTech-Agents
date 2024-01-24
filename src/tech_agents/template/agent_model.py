import langchain
from langchain.chains.llm import LLMChain
from langchain_openai import AzureChatOpenAI
from langchain.memory import ReadOnlySharedMemory, ConversationBufferMemory
from langchain.agents import BaseSingleActionAgent,  Tool, AgentType, initialize_agent, AgentExecutor
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

from . import default_value


# プロンプトの定義
# 日本語ver
# ROUTER_TEMPLATE = '''あなたの仕事は、以下の候補からユーザーの対応を任せるのに最適な選択肢を選び、その名前を回答することです。直接ユーザーへの回答は行わず、適切な候補を選ぶだけです。

# << 選択候補 >>
# 名前: 説明
# {destinations}

# << 出力形式 >>
# 選択した候補の名前のみを出力してください。全ての候補が不適切である場合は "DEFAULT" と回答してください。

# << 回答例 >>
# Human: 「あなたに与えられた役割はなんですか？」
# AI: "DEFAULT"
# '''

# 英語ver(トークン節約のため)
ROUTER_TEMPLATE = '''Your job is to select the best option from the candidates below to entrust the user to respond to the user and answer to the name. You do not respond directly to the user, only select the appropriate candidate.

# Candidate Selection
Name: Description.
{destinations}

# output format
Output only the names of the selected candidates. If all candidates are inappropriate, answer "DEFAULT".

# Sample Responses
Human: "What is your assigned role?"
AI: "DEFAULT"

# history
'''

# 追いプロンプトの定義
ROUTER_PROMPT_SUFFIX = '''
# Output Format Specification
I'll reiterate the instructions one last time. Please output only the name of the candidate you have selected.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT."
'''



class DestinationOutputParser(BaseOutputParser[str]):
    """
    このクラスは、ルーターチェーンの出力を解析して目的地を決定するための出力パーサーです。
    """

    destinations: Set[str]

    class Config:
        # 追加の設定を許可します。
        extra = Extra.allow

    def __init__(self, **kwargs):
        # 親クラスの初期化メソッドを呼び出します。
        super().__init__(**kwargs)
        # 目的地のリストに "DEFAULT" を追加します。
        self.destinations_and_default = list(self.destinations) + ["DEFAULT"]

    def parse(self, text: str) -> str:
        # 入力テキストが各目的地に含まれるかどうかをチェックします。
        matched = [int(d in text) for d in self.destinations_and_default]
        # マッチした目的地が1つだけでなければ、例外をスローします。
        if sum(matched) != 1:
            raise OutputParserException(
                f"DestinationOutputParser expected output value includes "
                f"one(and only one) of {self.destinations_and_default}. "
                f"Received {text}."
            )
        # マッチした目的地を返します。
        return self.destinations_and_default[matched.index(1)]

    @property
    def _type(self) -> str:
        # パーサーのタイプを返します。
        return "destination_output_parser"


class DispatcherAgent(BaseSingleActionAgent):
    """
    このクラスは、ユーザーの入力を受け取り、適切なツールを選択して実行するディスパッチャーエージェントです。
    """
    chat_model: BaseChatModel
    readonly_memory: ReadOnlySharedMemory
    tools: List[Tool]
    verbose: bool = False

    class Config:
        # 追加の設定を許可します。
        extra = Extra.allow

    def __init__(self, **kwargs):
        # 親クラスの初期化メソッドを呼び出します。
        super().__init__(**kwargs)
        # ツールのリストから各ツールの名前と説明を取得し、それらを改行で結合した文字列を作成します。
        destinations = "\n".join(
            [f"{tool.name}: {tool.description}" for tool in self.tools])
        # ルーターテンプレートを作成します。
        router_template = ROUTER_TEMPLATE.format(destinations=destinations)
        # チャットプロンプトテンプレートを作成します。
        router_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                template=router_template),
            MessagesPlaceholder(variable_name='chat_history'),
            HumanMessagePromptTemplate(prompt=PromptTemplate(
                input_variables=['input'], template='{input}')),
            SystemMessagePromptTemplate.from_template(
                template=ROUTER_PROMPT_SUFFIX)
        ])
        # ルーターチェーンを作成します。
        self.router_chain = LLMChain(
            llm=self.chat_model,
            prompt=router_prompt_template,
            memory=self.readonly_memory,
            verbose=self.verbose
        )
        # ルートパーサーを作成します。
        self.route_parser = DestinationOutputParser(
            destinations=set([tool.name for tool in self.tools])
        )

    @property
    def input_keys(self):
        # 入力キーを返します。
        return ["input"]

    def plan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        # ルーターチェーンを実行し、その出力を解析して目的地を決定します。
        router_output = self.router_chain.run(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException as ope:
            # 出力が解析できない場合、デフォルトの目的地が選択されます。
            destination = "DEFAULT"
        # 選択されたツールと入力、および空のログを含む`AgentAction`オブジェクトを返します。
        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")

    async def aplan(
        self, intermediate_steps: List[Tuple[AgentAction, str]], **kwargs: Any
    ) -> Union[AgentAction, AgentFinish]:
        # ルーターチェーンを非同期に実行し、その出力を解析して目的地を決定します。
        router_output = await self.router_chain.arun(kwargs["input"])
        try:
            destination = self.route_parser.parse(router_output)
        except OutputParserException as ope:
            # 出力が解析できない場合、デフォルトの目的地が選択されます。
            destination = "DEFAULT"
        # 選択されたツールと入力、および空のログを含む`AgentAction`オブジェクトを返します。
        return AgentAction(tool=destination, tool_input=kwargs["input"], log="")


class BaseDispatcherAgent:
    """
    このクラスは、ユーザーの入力を受け取り、適切なツールを選択して実行するディスパッチャーエージェントの基底クラスです。
    このクラスを継承して、ツールの定義を実装してください。
    
    --------------------
    実装方法:
    1. クラスの初期化メソッドで、DispatcherAgentの初期化を行う。
    ```
    class DispatcherAgent(BaseDispatcherAgent):
        def __init__(self, llm, memory, readonly_memory, chat_history, verbose):
            super().__init__(llm, memory, readonly_memory, chat_history, verbose)
            
        def define_tools(self) -> List[Tool]:
            ...
    ```
    2. define_tools メソッドで、ツールの定義を行う。
    ```
        def define_tools(self) -> List[Tool]:
            tool_1 = # 呼び出したいツールの定義１
            tool_2 = # 呼び出したいツールの定義２
            ...
            tools = [
                Tool.from_function(
                    func=tool_1.run, # ツールの実行関数
                    name="tool_1", # ツールの名前
                    description="tool_1の説明"
                    args_schema=tool_1_input_schema, # ツールの入力スキーマ
                    return_direct=True # ツールの出力を直接返すかどうか
                ), 
                Tool.from_function(
                    func=tool_2.run,
                    name="tool_2",
                    description="tool_2の説明"
                    args_schema=tool_2_input_schema,
                    return_direct=True
                )
                ...
            ]
            return tools
    ```
    3. run メソッドで、ツールの実行を行う。
    """
    def __init__(
        self, 
        llm: AzureChatOpenAI = default_value.default_llm,
        memory: ConversationBufferMemory = default_value.default_memory,
        readonly_memory: ReadOnlySharedMemory = default_value.default_readonly_memory,
        chat_history: MessagesPlaceholder = default_value.default_chat_history,
        verbose: bool = False,
        ):
        """
        このクラスは、ユーザーの入力を受け取り、適切なツールを選択して実行するディスパッチャーエージェントの基底クラスです。
        """
        self.llm = llm
        self.memory = memory
        self.readonly_memory = readonly_memory
        self.chat_history = chat_history
        self.verbose = verbose
        self.tools = self.define_tools()
        self.dispatcher_agent = self.create_dispatcher_agent()

    def define_tools(self) -> List[Tool]:
        """
        このメソッドは、ツールの定義を行います。
        --------------------
        実装方法:
        1. ツールのリストを作成する。
        2. ツールの定義を行う。
        3. ツールのリストを返す。
        """
        # ツールの定義をサブクラスで実装
        raise NotImplementedError("This method should be implemented by subclasses.")

    def create_dispatcher_agent(self) -> DispatcherAgent:
        return DispatcherAgent(
            chat_model=self.llm,
            readonly_memory=self.readonly_memory,
            tools=self.tools,
            verbose=self.verbose
        )

    def run(self, user_message: str) -> str:
        """
        `DispatcherAgent`の実行メソッドです。
        --------------------
        実装方法:
        ```
        return_message: str = dispatcher_agent.run(user_message: str) 
        ```
        """
        # 共通の run メソッド
        try:
            agent = AgentExecutor.from_agent_and_tools(
                agent=self.dispatcher_agent, tools=self.tools, memory=self.memory, verbose=self.verbose
            )
            return agent.run(user_message)
        except Exception as e:
            raise e



class BaseToolAgent:
    """
    このクラスは、ツールエージェントの基底クラスです。
    このクラスを継承して、ツールエージェントの定義を実装してください。
    --------------------
    実装方法:
    1. クラスの初期化メソッドで、ツールエージェントの初期化を行う。
    ```
    class ToolAgent(BaseToolAgent):
        def __init__(self, llm, memory, chat_history, verbose):
            super().__init__(llm, memory, chat_history, verbose)
            
        def run(self, input) -> str:
            ...
            return agent.run(input)
    ```
    """
    def __init__(
        self,
        llm: AzureChatOpenAI = default_value.default_llm,
        memory: ConversationBufferMemory = default_value.default_memory,
        chat_history: MessagesPlaceholder = default_value.default_chat_history,
        verbose: bool = False,
        model_kwargs: dict = None
        ):
        if model_kwargs: # モデルのkwargsを上書きする場合
            self.llm = AzureChatOpenAI(
                openai_api_base=llm.openai_api_base,
                openai_api_version=llm.openai_api_version,
                deployment_name=llm.deployment_name,
                openai_api_key=llm.openai_api_key,
                openai_api_type=llm.openai_api_type,
                temperature=llm.temperature,
                model_kwargs=model_kwargs
            )
        else:
            self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose
        langchain.debug = self.verbose

    def run(self, input) -> str:
        raise NotImplementedError(
            "This method should be implemented by subclasses.")

    def initialize_agent(
        self,
        agent_type: AgentType,
        tools: List,
        system_message_template: str
        ) -> initialize_agent:
        # エージェントの初期化
        agent_kwargs = {
            "system_message": SystemMessagePromptTemplate.from_template(template=system_message_template),
            "extra_prompt_messages": [self.chat_history]
        }
        agent_function = initialize_agent(
            tools=tools,
            llm=self.llm,
            agent=agent_type,
            verbose=self.verbose,
            agent_kwargs=agent_kwargs,
            memory=self.memory
        )
        return agent_function
