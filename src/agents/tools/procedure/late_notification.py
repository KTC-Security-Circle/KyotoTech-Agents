import os
from dotenv import load_dotenv
load_dotenv()

from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType, initialize_agent
import langchain
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.tools import tool
import json
import datetime
from pydantic.v1 import BaseModel, Field

verbose = True
langchain.debug = verbose



# 以下は授業名が「python機械学習」で担当講師が「木本」の場合の first_period_class の設定例です。

# ```
# {class_name: "python機械学習", instructor_name: "木本"}
# ```

# システムプロンプトの設定
LATE_NOTIFICATION_ITEMS_SYSTEM_PROMPT = '''あなたは生徒からの遅延届の申請を受け付ける担当者です。
遅延届とは、学校に利用電鉄が遅延し遅刻・欠席した場合に提出する書類です。
遅延届の申請を受け付けるには以下の情報が全て必要です。


# 必要な情報の項目
- 遅延した日
- 遅延による遅刻・欠席時限
- 上記時限の入室時刻
- 遅刻・欠席科目
- 担当講師名
- 利用電鉄会社
- 路線名
- 遅延していた時間
- 電鉄がWEB上に掲載する遅延証明内容と相違がないか。


# あなたの取るべき行動
- 必要な情報に未知の項目がある場合は予測や仮定をせず、"***" に置き換えた上で、把握している情報を late_notification_items 関数に設定し confirmed = false で実行して下さい。
- あなたの「電鉄がWEB上に掲載する遅延証明内容と相違はありませんか?」の問いかけに対して、ユーザーから肯定的な返答が確認できた場合のみ late_notification_items 関数を check_late_time = true で実行し申請を行って下さい。
- あなたの「最終確認です。以下の内容で遅延届を申請しますが、よろしいですか?」の問いかけに対して、ユーザーから肯定的な返答が確認できた場合のみ late_notification_items 関数を confirmed = true で実行し申請を行って下さい。
- ユーザーから手続きをやめる、キャンセルする意思を伝えられた場合のみ、 late_notification_items 関数を canceled = true で実行し、あなたはそれまでの公欠届の申請に関する内容を全て忘れます。

# 重要な注意事項
初期値は全て "***" です。
必要な情報に未知の項目がある場合は予測や仮定をせず "***" に置き換えてください。
ユーザーから与えられた情報以外は使用せず、想像で補完しないでください。

late_notification_items 関数はユーザーから遅延届の申請の手続きをやめる、キャンセルする意思を伝えられた場合のみ canceled = true で実行して、
それまでの遅延届の申請に関する内容を全て忘れてください。

late_notification_items 関数は次に示す例外を除いて confirmed = false で実行してください。

あなたの「最終確認です。以下の内容で遅延届を申請しますが、よろしいですか?」の問いかけに対して、
ユーザーから肯定的な返答が確認できた場合のみ late_notification_items 関数を confirmed = true で実行して部品を注文してください。

最終確認に対するユーザーの肯定的な返答なしで late_notification_items 関数を confirmed = true で実行することは誤申請であり事故になるので、固く禁止します。

'''




class LateNotificationItemsInput(BaseModel):
    date: str = Field(
        description="遅延した日付です。形式は'2023/12/25'のような'年/月/日'の形式です。")
    late_class: str = Field(description="遅延による遅刻・欠席した授業の時限です。形式は'1限'のような'数値限'の形式です。")
    in_class_time: str = Field(description="教室に入室した時間です。形式は'10:00'のような'時:分'の形式です。")
    late_class_name: str = Field(description="遅延による遅刻・欠席した授業の名称です。")
    late_class_instructor: str = Field(description="遅延による遅刻・欠席した授業の担当講師名です。")
    use_public_transportation: str = Field(description="利用した電鉄会社の名称です。")
    use_transportation_name: str = Field(description="利用した電鉄の路線の名称です。")
    late_time: str = Field(description="遅延した時間です。形式は'30分'のような'数値分'の形式です。")
    check_late_time: bool = Field(description=(
        "電鉄がWEB上に掲載する遅延証明内容と相違がないかの確認状況です。\n"
        "相違がない場合は True, そうでなければ False としてください。\n"
        "* check_late_time が False の場合はユーザーに確認をとります。 \n"
        "* check_late_time が True の場合は遅延証明内容の確認が行われた証明です。")
    )
    confirmed: bool = Field(description=(
        "注文内容の最終確認状況です。最終確認が出来ている場合は True, そうでなければ False としてください。\n"
        "* confirmed が True の場合は部品の注文が行われます。 \n"
        "* confirmed が False の場合は注文内容の確認が行われます。")
    )
    canceled: bool = Field(description=(
        "注文の手続きを継続する意思を示します。\n"
        "通常は False としますがユーザーに注文の手続きを継続しない意図がある場合は True としてください。\n"
        "* canceled が False の場合は部品の注文手続きを継続します。 \n"
        "* canceled が True の場合は注文手続きをキャンセルします。")
    )


@tool("late_notification_items", return_direct=True, args_schema=LateNotificationItemsInput)
def late_notification_items(
    date: str,
    late_class: str,
    in_class_time: str,
    late_class_name: str,
    late_class_instructor: str,
    use_public_transportation: str,
    use_transportation_name: str,
    late_time: str,
    check_late_time: bool,
    confirmed: bool,
    canceled: bool,
) -> str:
    """遅延届の申請を行う関数です。"""
    if canceled:
        return "わかりました。また各種申請が必要になったらご相談ください。"

    def check_params(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time):
        for arg in [date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time]:
            if arg is None or arg == "***" or arg == "":
                return False
        return True

    has_required = check_params(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time)
    

    # 注文情報のテンプレート
    response_template = (
        f'・遅延した日: {date}\n'
        f'・遅延による遅刻・欠席時限: {late_class}\n'
        f'・上記時限の入室時刻: {in_class_time}\n'
        f'・遅刻・欠席科目: {late_class_name}\n'
        f'・担当講師名: {late_class_instructor}\n'
        f'・利用電鉄会社: {use_public_transportation}\n'
        f'・路線名: {use_transportation_name}\n'
        f'・遅延していた時間: {late_time}\n'
        f'・電鉄がWEB上に掲載する遅延証明内容と相違がないか。: {"確認済み" if check_late_time else "未確認"}\n'
    )

    # 追加情報要求のテンプレート
    request_information_template = (
        f'申請には以下の情報が必要です。"***" の項目を教えてください。\n'
        f'\n'
        f'{response_template}'
    )
    
    # 遅延証明確認のテンプレート
    check_template = (
        f'電鉄がWEB上に掲載する遅延証明内容と相違はありませんか?\n'
        f'\n{response_template}'
    )

    # 注文確認のテンプレート
    confirm_template = (
        f'最終確認です。以下の内容で遅延届を申請しますが、よろしいですか?\n'
        f'\n{response_template}'
    )

    # 注文完了のテンプレート
    def request_late_notification(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time):
        try:
            t_delta = datetime.timedelta(hours=9)
            JST = datetime.timezone(t_delta, 'JST')
            now = datetime.datetime.now(JST)
            date_time = now.strftime('%Y%m%d%H%M%S')
            file_dir = f'{os.path.dirname(os.path.abspath(__file__))}/late_notification/{date_time}.json'
            with open(file_dir, "w", encoding='utf-8') as f:
                late_notification_template = {'late_notification': {
                    'adsence_time': date_time, 'date': date, 'late_class': late_class, 'in_class_time': in_class_time, 'late_class_name': late_class_name, 'late_class_instructor': late_class_instructor, 'use_public_transportation': use_public_transportation, 'use_transportation_name': use_transportation_name, 'late_time': late_time, 'check_late_time': "確認済み"}}
                f.write(json.dumps(late_notification_template,
                        indent=4, ensure_ascii=False))
            response = (
                f'遅延届を以下の内容で申請しました。\n'
                f'\n{response_template}'
                f'\n学生ポータルサイトの各種申請詳細に今回の申請内容が申請されていない場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        except:
            response = (
                f'遅延届の申請に失敗しました。\n'
                f'お時間をおいてからも再度失敗する場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        return response

    if has_required and confirmed and check_late_time:
        return request_late_notification(date, late_class, in_class_time, late_class_name, late_class_instructor, use_public_transportation, use_transportation_name, late_time)
    else:
        if has_required and check_late_time:
            return confirm_template
        elif has_required and not check_late_time:
            return check_template
        else:
            return request_information_template


late_notification_items_tools = [late_notification_items]



# モデルの初期化
# llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
#     openai_api_base=os.environ["OPENAI_API_BASE"],
#     openai_api_version=os.environ["OPENAI_API_VERSION"],
#     deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
#     openai_api_key=os.environ["OPENAI_API_KEY"],
#     openai_api_type="azure",
#     model_kwargs={"top_p": 0.1, "function_call": {
#         "name": "late_notification_items"}}
# )

def test(input, verbose, memory, chat_history, llm):
    test_llm = llm.copy()
    test_llm.model_kwargs = {"top_p": 0.1, "function_call": {"name": "late_notification_items"}}
    print(test_llm)
    print(verbose)
    print(memory)
    print(chat_history)

def run(input, verbose, memory, chat_history, llm):
    late_notification_llm = llm.copy()
    late_notification_llm.model_kwargs = {"top_p": 0.1, "function_call": {"name": "late_notification_items"}}
    agent_kwargs = {
        "system_message": SystemMessagePromptTemplate.from_template(template=LATE_NOTIFICATION_ITEMS_SYSTEM_PROMPT),
        "extra_prompt_messages": [chat_history]
    }
    late_notification_agent = initialize_agent(
        late_notification_items_tools,
        late_notification_llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=verbose,
        agent_kwargs=agent_kwargs,
        memory=memory
    )
    res = late_notification_agent.run(input)
    return res

# message = "公欠届を申請したいです。"
# print(official_absence_agent.run(message))

# while True:
#     message = input(">> ")
#     if message == "exit":
#         break
#     response = late_notification_agent.run(message)
#     print(response)
