from pydantic.v1 import BaseModel, Field
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, ChatPromptTemplate


prompt = ChatPromptTemplate.from_messages([
    ("system", "Summarize the text you give next into a form that fits the user's question. summary: {summary}"),
    ("human", "{input}")
])



class DDGSearchInput(BaseModel):  # 検索ワードを入力するためのモデルを作成。
    search_word: str = Field(description="ユーザーからの入力から生成される検索ワードです。")



class DDGSearchAgent:
    def __init__(self, llm, memory, chat_history, verbose):
        self.llm = llm
        self.memory = memory
        self.chat_history = chat_history
        self.verbose = verbose
        
    def summary(self, summary, input):
        chain = prompt | self.llm
        inputs = {"summary": summary, "input": input}
        return chain.invoke(inputs).content
    
    def ddg_search(self, input):
        ddg_search = DuckDuckGoSearchRun()
        try:
            return ddg_search.run(input)
        except Exception as e:
            print(e)
            return f'検索に失敗しました。時間をおいてから再度お試しください。'
    
    def run(self, input):
        search_text = self.ddg_search(input)
        if search_text == '検索に失敗しました。時間をおいてから再度お試しください。':
            return search_text
        else:
            return self.summary(summary=search_text, input=input)
