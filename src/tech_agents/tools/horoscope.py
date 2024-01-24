from langchain.agents import AgentType, tool
import json
import requests
import datetime
from pydantic.v1 import BaseModel, Field

from ..template.agent_model import BaseToolAgent

# システムプロンプトの設定
# HOROSCOPE_SYSTEM_PROMPT = '''あなたは星占いの専門家です。
# 星占いをしてその結果を回答します。

# ただし星占いには誕生日が必要です。
# もし誕生日が分からない場合は、誕生日を予測や仮定をせずに「星占いをするので誕生日を教えてください。」と回答して下さい。

# 誕生日がわかる場合は、例えば"4月24日"であれば"04/24"の形式に変換した上で horoscope 関数を使って占いを行って下さい。
# '''
HOROSCOPE_SYSTEM_PROMPT = '''You are an expert in astrology and will provide horoscopes as a response.

However, a birthday is necessary for astrology.
If the birthday is unknown, please respond without making predictions or assumptions, asking, 'Please tell me your birthday for the horoscope.'

If the birthday is known, for example, if it is 'April 24th,' please convert it into the format '04/24' and use the horoscope function to conduct the astrology reading.

Respond in Japanese.
'''


# エージェントの初期化
class HoroscopeInput(BaseModel) : # 誕生日を入力するためのモデルを作成。
    birthday: str = Field(
        description="'mm/dd'形式の誕生日です。例: 3月7日生まれの場合は '03/07' です。")

@tool("horoscope", return_direct=True, args_schema=HoroscopeInput) # Agentsツールを作成。
def horoscope(birthday: str): # 誕生日を入力すると、星占いをしてくれる関数を作成。
    """星占いで今日の運勢を占います。"""
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



horoscope_tools = [horoscope]


class HoroscopeAgentInput(BaseModel):
    user_utterance: str = Field(
        description="This is the user's most recent utterance that is communicated to the astrologer.")



class HoroscopeAgent(BaseToolAgent):
    def __init__(self, llm, memory, chat_history, verbose):
        super().__init__(llm, memory, chat_history, verbose)
        # HoroscopeAgent 特有の初期化（もしあれば）

    def run(self, input):
        # HoroscopeAgent特有の処理
        horoscope_agent = self.initialize_agent(
            agent_type=AgentType.OPENAI_FUNCTIONS,
            tools=horoscope_tools,  # 事前に定義されたhoroscope関数
            system_message_template=HOROSCOPE_SYSTEM_PROMPT
        )
        return horoscope_agent.run(input)
