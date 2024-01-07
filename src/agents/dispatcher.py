import os

from langchain.chat_models import AzureChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
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

from agents.tools import (
    horoscope,
    search,
    procedure,
    default
)


# Azure OpenAIのAPIを読み込み。
default_llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
    temperature=0,
    model_kwargs={"top_p": 0.1}
)



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


class DestinationOutputParser(BaseOutputParser[str]): # 出力パーサーを作成。
    destinations: Set[str]

    class Config: # 出力パーサーの設定。
        extra = Extra.allow

    def __init__(self, **kwargs): # 出力パーサーの初期化。
        super().__init__(**kwargs)
        self.destinations_and_default = list(self.destinations) + ["DEFAULT"]

    def parse(self, text: str) -> str: # 
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




class MainAgent:
    '''メインエージェントのクラスです。'''

    def __init__(self, llm, memory, chat_history, verbose):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose
        
        self.default_agent = default.Agent(llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def default_agent_wrapper(user_message):
            return self.default_agent.run(user_message)
        self.horoscope_agent = horoscope.Agent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def horoscope_agent_wrapper(user_message):
            return self.horoscope_agent.run(user_message)
        self.search_database_agent = search.Agent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def search_database_agent_wrapper(user_message):
            return self.search_database_agent.run(user_message)
        self.procedure_agent = procedure.Agent(
            llm=self.llm, memory=self.memory, chat_history=self.chat_history, verbose=self.verbose)
        def procedure_agent_wrapper(user_message):
            return self.procedure_agent.run(user_message)

        class HoroscopeAgentInput(BaseModel):
            user_utterance: str = Field(
                description="This is the user's most recent utterance that is communicated to the astrologer.")

        class SearchDBAgentInput(BaseModel):
            user_utterance: str = Field(
                description="The user's most recent utterance that is communicated to the person in charge of the school database search.")

        class ProcedureAgentInput(BaseModel):
            user_utterance: str = Field(
                description="This is the user's most recent utterance that is communicated to the person in charge of various procedures.")

        class DefaultAgentInput(BaseModel):
            user_utterance: str = Field(
                description="This is the user's most recent utterance that communicates general content to the person in charge.")

        self.tools = [
            Tool.from_function(
                func=horoscope_agent_wrapper,
                name="horoscope",
                description="This is the person in charge of astrology. This person should be in charge of handling conversations related to horoscopes.",
                args_schema=HoroscopeAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=search_database_agent_wrapper,
                name="search_database",
                description="This person is in charge of school database searches. This person should be responsible for searching the school database and handling conversations related to school information.",
                args_schema=SearchDBAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=procedure_agent_wrapper,
                name="procedure",
                description="This person is in charge of various procedures. This person should be responsible for handling conversations related to various procedures.",
                args_schema=ProcedureAgentInput,
                return_direct=True
            ),
            Tool.from_function(
                func=default_agent_wrapper,
                name="DEFAULT",
                description="This is the person in charge of general conversations. This person should be assigned to handle conversations that are general and should not be left to a specific expert.",
                args_schema=DefaultAgentInput,
                return_direct=True
            ),
        ]
    
    
    def run(self, user_message: str):
        dispatcher_agent = DispatcherAgent(
            chat_model=default_llm, readonly_memory=self.memory, tools=self.tools, verbose=self.verbose)
        agent = AgentExecutor.from_agent_and_tools(
            agent=dispatcher_agent, tools=self.tools, memory=self.memory, verbose=self.verbose
        )
        return agent.run(user_message)

