import os
import typing

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType, initialize_agent, tool
import langchain

from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate
import json
import requests
import datetime
from langchain.tools import tool
from pydantic import BaseModel, Field

# モデルの初期化
llm = AzureChatOpenAI( # Azure OpenAIのAPIを読み込み。
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
)


# システムプロンプトの設定
HOROSCOPE_SYSTEM_PROMPT = '''あなたは星占いの専門家です。
星占いをしてその結果を回答します。

ただし星占いには誕生日が必要です。
もし誕生日が分からない場合は、誕生日を予測や仮定をせずに「星占いをするので誕生日を教えてください。」と回答して下さい。

誕生日がわかる場合は、例えば"4月24日"であれば"04/24"の形式に変換した上で horoscope 関数を使って占いを行って下さい。
'''



# エージェントの初期化
class HoroscopeInput(BaseModel) : # 誕生日を入力するためのモデルを作成。
    birthday: str = Field(
        description="'mm/dd'形式の誕生日です。例: 3月7日生まれの場合は '03/07' です。")

@tool("horoscope", return_direct=True, args_schema=HoroscopeInput) # Agentsツールを作成。
def horoscope(input_data: HoroscopeInput): # 誕生日を入力すると、星占いをしてくれる関数を作成。
    """星占いで今日の運勢を占います。"""
    birthday = input_data.birthday
    birthday = "02/28" if birthday == "02/29" else birthday
    yday = datetime.datetime.strptime(birthday, '%m/%d').timetuple().tm_yday
    sign_table = {
        20: '山羊座', 50: '水瓶座', 81: '魚座', 111: '牡羊座',  142: '牡牛座',
        174: '双子座', 205: '蟹座', 236: '獅子座',  267: '乙女座', 298: '天秤座',
        328: '蠍座', 357: '射手座', 999: '山羊座',
    }
    for k, v in sign_table.items():
        if yday < k:
            sign = v
            break

    t_delta = datetime.timedelta(hours=9)
    JST = datetime.timezone(t_delta, 'JST')
    today = datetime.datetime.now(JST).strftime('%Y/%m/%d')
    url = f"http://api.jugemkey.jp/api/horoscope/free/{today}"
    response = requests.get(url)
    horoscope = json.loads(response.text)["horoscope"][today]
    horoscope = {h["sign"]: h for h in horoscope}
    horoscope[sign]
    content = \
    f'''今日の{sign}の運勢は...
    ・{horoscope[sign]["content"]}
    ・ラッキーアイテム:{horoscope[sign]["item"]}
    ・ラッキーカラー:{horoscope[sign]["color"]}'''
    return content






verbose = True
langchain.debug = verbose

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
chat_history = MessagesPlaceholder(variable_name='chat_history')

horoscope_tools = [horoscope]

agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(content=HOROSCOPE_SYSTEM_PROMPT),
    "extra_prompt_messages": [chat_history]
}
horoscope_agent = initialize_agent(
    horoscope_tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=verbose,
    agent_kwargs=agent_kwargs,
    memory=memory
)


horoscope_agent.run("私の今日の運勢を教えて。")
