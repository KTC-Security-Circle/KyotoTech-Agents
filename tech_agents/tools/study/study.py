from langchain.agents import AgentType, tool
import json
import requests
import datetime
from pydantic.v1 import BaseModel, Field

from tech_agents.template.vector_search import search_vector
from tech_agents.template.agent_model import BaseToolAgent



# システムプロンプトの設定
STUDY_SYSTEM_PROMPT = '''あなたは授業のカリキュラムから学生の要望に応える教師AIで、学生の理解度を確認するために問題を作成し、回答に応じて適切に対応します。
間違った回答には、丁寧な解説と似た問題を再度提示して理解を深めます。
正解の場合は、少し難易度を上げた応用問題を出して、学生の学習を促進します。
このアプローチは、学生が自信を持ち、プログラミングの楽しさを感じるように設計されています。
複雑なトピックや他の言語には対応しません。
あなたは授業のカリキュラムを study という関数で確認することができます。
授業の内容やカリキュラムを確認したい場合は、確認したい内容を入力してください。
確認した内容から問題を作成したり、学生の質問に答えたりすることができます。
カリキュラムは授業を行う先生の情報をもとに作成されており、15回の授業で行う内容などが記載されています。

'''
STUDY_SYSTEM_PROMPT = '''You are a teacher AI that responds to student requests from the class curriculum, creating questions to check student understanding and responding appropriately to their answers.
For wrong answers, you provide careful explanations and re-present similar questions to deepen understanding.
For correct answers, application questions of slightly increased difficulty are offered to facilitate student learning.
This approach is designed to help students gain confidence and enjoy programming.
It does not address complex topics or other languages.
You can check your class curriculum with the function study.
If you want to check the content or curriculum of a class, enter the content you want to check.
You can create questions or answer student questions based on what you have reviewed.
The curriculum is based on the information of the teacher who teaches the class, and includes the contents of the 15 lessons.

Respond in Japanese.
'''


def search_database(study_input):
    docs = search_vector("vector-class-data", study_input)
    i = 1
    search_result = []
    for doc in docs:
        search_result.append(
            f'{i}件目の検索結果\n'
            f'授業名: {doc.metadata["class_name"]}\n'
            f'検索内容: {doc.page_content}\n'
        )
        i += 1
    return search_result

# エージェントの初期化
class StudyInput(BaseModel):  
    study_input: str = Field(
        description="データベースに検索するための入力です。")


@tool("study", return_direct=False, args_schema=StudyInput)  # Agentsツールを作成。
def study(study_input: str):  
    """授業データベースにアクセスし、検索結果を返答します。"""
    try:
        search_result = search_database(study_input)
    except Exception as e:
        return "検索に失敗しました。"
    if len(search_result) == 0:
        return "該当する内容は見つかりませんでした。"
    else:
        return "\n".join(search_result)


study_tools = [study]


class StudyAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the teacher.")


class StudyAgent(BaseToolAgent):
    def __init__(self, llm, memory, chat_history, verbose):
        super().__init__(llm, memory, chat_history, verbose)

    def run(self, input):
        study_agent = self.initialize_agent(
            agent_type=AgentType.OPENAI_FUNCTIONS,
            tools=study_tools,  
            system_message_template=STUDY_SYSTEM_PROMPT
        )
        return study_agent.run(input)
