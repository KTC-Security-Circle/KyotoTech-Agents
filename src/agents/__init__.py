import os

import langchain
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.prompts.chat import MessagesPlaceholder

from agents.main import MainAgent
from . import main, tools

def run(input: str):
    return MainAgent.run(input)
