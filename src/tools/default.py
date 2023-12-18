import os

from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

from grobal import grobal_value as g

DEFAULT_SYSTEM_PROMPT = '''あなたはAIのアシスタントです。
ユーザーの質問に答えたり、議論したり、日常会話を楽しんだりします。
'''

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(template=DEFAULT_SYSTEM_PROMPT),
    g.chat_history,
    HumanMessagePromptTemplate(prompt=PromptTemplate(
        input_variables=['input'], template='{input}'))
])

default_chain = LLMChain(llm=g.llm, prompt=chat_prompt_template, memory=g.readonly_memory, verbose=g.verbose)


# debag
# print(default_chain.run("あなたは誰ですか？"))