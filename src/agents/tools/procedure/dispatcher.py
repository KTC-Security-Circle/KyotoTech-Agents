import os

import langchain
from langchain.chains.llm import LLMChain
from langchain.memory import ReadOnlySharedMemory
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
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory


import late_notification
import official_absence


verbose = True
langchain.debug = verbose

# Azure OpenAIのAPIを読み込み。
llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
    temperature=0,
    model_kwargs={"top_p": 0.1}
)

# 会話メモリの定義
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
readonly_memory = ReadOnlySharedMemory(memory=memory)
chat_history = MessagesPlaceholder(variable_name='chat_history')


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
    destinations: Set[str]

    class Config:
        extra = Extra.allow

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.destinations_and_default = list(self.destinations) + ["DEFAULT"]

    def parse(self, text: str) -> str:
        matched = [int(d in text) for d in self.destinations_and_default]
        if sum(matched) != 1:
            raise OutputParserException(
                f"DestinationOutputParser expected output value includes "
                f"one(and only one) of {self.destinations_and_default}. "
                f"Received {text}."
            )

        return self.destinations_and_default[matched.index(1)]

    @property
    def _type(self) -> str:
        return "destination_output_parser"


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


def defalt_answer():
    res = (
        f'私が行うことのできる各種申請は以下の通りです。\n'
        f'・公欠届\n'
        f'・遅延届\n'
        f'○○を申請したいと言ってもらえれば、詳細を聞き申請をすることができます。'
    )
    return res

def late_notification_agent(input):
    return late_notification.run(input=input, verbose=verbose, memory=memory, chat_history=chat_history, llm=llm)

def official_absence_agent(input):
    return official_absence.run(input=input, verbose=verbose, memory=memory, chat_history=chat_history, llm=llm)


class LateNotificationAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the person in charge of delay notification application")

# class PartsOrderAgentInput(BaseModel):
#     user_utterance: str = Field(
#         description="プラモデルの部品の個別注文の担当者に伝達するユーザーの直近の発話内容です。")


class OfficialAbsenceAgentInput(BaseModel):
    user_utterance: str = Field(
        description="The user's most recent utterance that is communicated to the person in charge of application for official absence notification")





tools = [
    Tool.from_function(
        func=late_notification_agent,
        name="late_notification",
        description="遅延届の申請に関する担当者です。遅延届に関係する会話の対応はこの担当者に任せるべきです。",
        args_schema=LateNotificationAgentInput,
        return_direct=True
    ),
    Tool.from_function(
        func=official_absence_agent,
        name="official_absence",
        description="公欠届の申請に関する担当者です。公欠届に関係する会話の対応はこの担当者に任せるべきです。",
        args_schema=OfficialAbsenceAgentInput,
        return_direct=True
    ),
    Tool.from_function(
        func=defalt_answer,
        name="DEFAULT",
        description="特定の申請に関する情報がない場合はこの担当者に任せるべきです。",
        return_direct=True
    ),
]




dispatcher_agent = DispatcherAgent(
    chat_model=llm, readonly_memory=readonly_memory, tools=tools, verbose=verbose)
agent = AgentExecutor.from_agent_and_tools(
    agent=dispatcher_agent, tools=tools, memory=memory, verbose=verbose
)



# def run(input: str):
#     return agent.run(input)

# while(True):
#     message = input(">> ")
#     if message == "exit" or message == ":q":
#         break
#     try:
#         agent.run(message)
#     except Exception as e:
#         print(e)

message = "公欠届を申請したいです。"
print(late_notification(message))

