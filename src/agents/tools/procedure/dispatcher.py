import langchain
from langchain_openai import AzureChatOpenAI
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
from pydantic.v1 import Extra, BaseModel, Field
from typing import Any, List, Tuple, Set, Union
from langchain.memory import ReadOnlySharedMemory

from ...template import default_value
from ...tools import procedure



# プロンプトの定義
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

# 英語ver(トークン節約のため)
ROUTER_TEMPLATE = '''Your job is to read the conversation between the user and yourself, and based on the descriptions provided below, select the most suitable candidate to handle the user's response.
You should not directly answer the user; your role is solely to choose the appropriate candidate.

<< Choices >>
Name: Description
{destinations}

<< Output Format >>
Please output only the name of the selected candidate.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT."

<< Example Answer >>
If asked, 'Tell me about yourself,' you should not respond.
Since there is no appropriate candidate in the choices, answer with "DEFAULT.
'''

ROUTER_PROMPT_SUFFIX = '''<< Output Format Specification >>
I'll reiterate the instructions one last time. Please output only the name of the candidate you have selected.
Note: The output must always be one of the names listed as choices. However, if you determine that all provided choices are inappropriate, you may use "DEFAULT."
'''



class DestinationOutputParser(BaseOutputParser[str]):
    # 目的地の集合を定義します。
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
    # チャットモデル、読み取り専用メモリ、ツールのリスト、および詳細なログ出力を制御するブール値フラグを定義します。
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


class ProcedureAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")


class ProcedureAgent:

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
        
        langchain.debug = self.verbose
        
        self.defalt_answer = (
            f'私が行うことのできる各種申請は以下の通りです。\n'
            f'・公欠届\n'
            f'・遅延届\n'
            f'○○を申請したいと言ってもらえれば、詳細を聞き申請をすることができます。'
        )
        def defalt_answer_wrapper(input):
            return self.defalt_answer
        self.late_notification_agent = procedure.LateNotificationAgent(llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def late_notification_agent_wrapper(input):
            return self.late_notification_agent.run(input)
        self.official_absence_agent = procedure.OfficialAbsenceAgent(llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def official_absence_agent_wrapper(input):
            return self.official_absence_agent.run(input)




        self.tools = [ # ツールのリスト
            Tool.from_function(
                func=late_notification_agent_wrapper,
                name="late_notification",
                # description="遅延届の申請に関する担当者です。遅延届に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is the person in charge regarding the late notification application. This person should be the person in charge of handling conversations related to the late notification.",  # 英語ver
                args_schema=procedure.LateNotificationAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=official_absence_agent_wrapper,
                name="official_absence",
                # description="公欠届の申請に関する担当者です。公欠届に関係する会話の対応はこの担当者に任せるべきです。", # 日本語ver
                description="This is the person in charge regarding the application for the Notification of Public Absence. This person should be the person in charge of handling conversations related to the public absence notification.", # 英語ver
                args_schema=procedure.OfficialAbsenceAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=defalt_answer_wrapper,
                name="DEFAULT",
                # description="特定の申請に関する情報がない場合はこの担当者に任せるべきです。", # 日本語ver
                description="If you do not have information on a particular application, this person should be assigned to you.", # 英語ver
                return_direct=True
            ),
        ]


    def run(self, input: str):
        self.dispatcher_agent = DispatcherAgent( # ディスパッチャーエージェントの初期化
            chat_model=self.llm, readonly_memory=self.readonly_memory, tools=self.tools, verbose=self.verbose)
        self.agent = AgentExecutor.from_agent_and_tools(
            agent=self.dispatcher_agent, tools=self.tools, memory=self.memory, verbose=self.verbose
        )
        return self.agent.run(input)


# while(True):
#     message = input(">> ")
#     if message == "exit" or message == ":q":
#         break
#     try:
#         output = agent.run(message)
#         print(output)
#     except Exception as e:
#         print(e)


