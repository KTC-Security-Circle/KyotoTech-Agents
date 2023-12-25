import os
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import AzureChatOpenAI
from langchain.agents import AgentType, initialize_agent
import langchain
from langchain.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate
from langchain.tools import tool
import json
import requests
import datetime
from typing import Union, List, Dict
from pydantic.v1 import BaseModel, Field

verbose = True
langchain.debug = verbose



# 以下は授業名が「python機械学習」で担当講師が「木本」の場合の first_period_class の設定例です。

# ```
# {class_name: "python機械学習", instructor_name: "木本"}
# ```

# システムプロンプトの設定
APPLICATION_ITEMS_SYSTEM_PROMPT = '''あなたは生徒からの公欠届の申請を受け付ける担当者です。
公欠届とは、学校に許可された都合で学校を休む場合に提出する書類です。
公欠届の申請を受け付けるには以下の情報が全て必要です。


# 必要な情報の項目
- 公欠日 :
- 欠席する授業の時限と名称、担当講師名 :
- 公欠事由 :


# あなたの取るべき行動
- 必要な情報に未知の項目がある場合は予測や仮定をせず、"***" に置き換えた上で、把握している情報を application_items 関数に設定し confirmed = false で実行して下さい。
- あなたの「最終確認です。以下の内容で公欠届を申請しますが、よろしいですか?」の問いかけに対して、ユーザーから肯定的な返答が確認できた場合のみ application_items 関数を confirmed = true で実行し申請を行って下さい。
- ユーザーから手続きをやめる、キャンセルする意思を伝えられた場合のみ、 application_items 関数を canceled = true で実行し、あなたはそれまでの公欠届の申請に関する内容を全て忘れます。

# application_items 関数を実行する際の欠席する授業の時限と名称の扱い
欠席する授業に関して、引数の first_period_class から fifth_period_class までの5つの時限の引数にそれぞれ設定してください。
場合によってはすべて入れる必要がない場合もあります。
first_period_class から fifth_period_class までの5つの引数の値は欠席する授業の名称とその講師名を持つ dict です。
dict の key として class_name と instructor_name を持ちます。
class_name の value が授業名の文字列、instructor_name の value が担当講師の文字列です。


'''
APPLICATION_ITEMS_SSUFFIX_PROMPT = '''

# 重要な注意事項
必要な情報に未知の項目がある場合は予測や仮定をせず "***" に置き換えてください。
ユーザーから与えられた情報以外は使用せず、想像で補完しないでください。

application_items 関数はユーザーから公欠届の申請の手続きをやめる、キャンセルする意思を伝えられた場合のみ canceled = true で実行して、
それまでの公欠届の申請に関する内容を全て忘れてください。

application_items 関数は次に示す例外を除いて confirmed = false で実行してください。

あなたの「最終確認です。以下の内容で公欠届を申請しますが、よろしいですか?」の問いかけに対して、
ユーザーから肯定的な返答が確認できた場合のみ application_items 関数を confirmed = true で実行して部品を注文してください。

最終確認に対するユーザーの肯定的な返答なしで application_items 関数を confirmed = true で実行することは誤申請であり事故になるので、固く禁止します。
'''


# エージェントの初期化
# class PeriodClass(BaseModel):



class ApplicationItemsInput(BaseModel):
    date: str = Field(
        description="公欠する日付です。形式は'2023/12/25'のような'年/月/日'の形式です。", default=None)
    first_period_class: dict[str, str] = Field(description=(
        "欠席する1時限目の Dict です。\n"
        "Dict は key class_name の value が授業名の文字列、key instructor_name の value が担当講師の文字列です。\n"
        "例: 授業名が'python機械学習'で担当講師が'木本'の場合は、\n"
        "\n"
        "{class_name: 'python機械学習', instructor_name: '木本'}\n"
        "\n"
        "としてください。"),
    )
    second_period_class: dict[str, str] = Field(
        description="欠席する2時限目の dict です。\n 形式は first_period_class と同じです。")
    third_period_class: dict[str, str] = Field(
        description="欠席する3時限目の dict です。\n 形式は first_period_class と同じです。",)
    fourth_period_class: dict[str, str] = Field(
        description="欠席する4時限目の dict です。\n 形式は first_period_class と同じです。",)
    fifth_period_class: dict[str, str] = Field(
        description="欠席する5時限目の dict です。\n 形式は first_period_class と同じです。",)
    reason: str = Field(description="公欠の理由です。")
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


@tool("application_items", return_direct=True, args_schema=ApplicationItemsInput)
def application_items(
    date: str,
    first_period_class: dict[str, str],
    second_period_class: dict[str, str],
    third_period_class: dict[str, str],
    fourth_period_class: dict[str, str],
    fifth_period_class: dict[str, str],
    reason: str,
    confirmed: bool,
    canceled: bool,
) -> str:
    """公欠届の申請を行う関数です。"""
    if canceled:
        return "わかりました。また各種申請が必要になったらご相談ください。"

    def check_params(date, first_period_class, second_period_class, third_period_class, fourth_period_class, fifth_period_class, reason):
        if date is None or date == "***" or date == "":
            return False
        if reason is None or reason == "***" or reason == "":
            return False
        for arg in [first_period_class, second_period_class, third_period_class, fourth_period_class, fifth_period_class]:
            if arg is None or arg == "***" or arg == "":
                return False
            if "class_name" not in arg or "instructor_name" not in arg:
                return False
            if arg["class_name"] == "***" or arg["instructor_name"] == "***":
                return False

        return True

    has_required = check_params(date, first_period_class, second_period_class,
                                third_period_class, fourth_period_class, fifth_period_class, reason)

    # 注文情報のテンプレート
    order_template = (
        f'・公欠日: {date}\n'
        f'・1限目: {first_period_class["class_name"]}/{first_period_class["instructor_name"]}\n'
        f'・2限目: {second_period_class["class_name"]}/{second_period_class["instructor_name"]}\n'
        f'・3限目: {third_period_class["class_name"]}/{third_period_class["instructor_name"]}\n'
        f'・4限目: {fourth_period_class["class_name"]}/{fourth_period_class["instructor_name"]}\n'
        f'・5限目: {fifth_period_class["class_name"]}/{fifth_period_class["instructor_name"]}\n'
        f'・公欠事由: {reason}\n'
    )

    # 追加情報要求のテンプレート
    request_information_template = (
        f'申請には以下の情報が必要です。"***" の項目を教えてください。\n'
        f'\n'
        f'{order_template}'
    )

    # 注文確認のテンプレート
    confirm_template = (
        f'最終確認です。以下の内容で公欠届を申請しますが、よろしいですか?\n'
        f'\n{order_template}'
    )

    # 注文完了のテンプレート
    def request_official_absence(date, first_period_class, second_period_class, third_period_class, fourth_period_class, fifth_period_class, reason):
        try:
            datetime = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            file_dir = f'{os.path.dirname(os.path.abspath(__file__))}/official_absence/{datetime}.json'
            with open(file_dir, "w") as f:
                official_absence_template = {'official_absence': {'adsence_time': datetime, 'date': date, 'first_period_class': first_period_class, 'second_period_class': second_period_class,
                                                                  'third_period_class': third_period_class, 'fourth_period_class': fourth_period_class, 'fifth_period_class': fifth_period_class, 'reason': reason}}
                f.write(json.dumps(official_absence_template, indent=4))
            response = (
                f'公欠届を以下の内容で申請しました。\n'
                f'\n{order_template}'
                f'\n学生ポータルサイトの各種申請詳細に今回の申請内容が申請されていない場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        except:
            response = (
                f'公欠届の申請に失敗しました。\n'
                f'お時間をおいてからも再度失敗する場合は、\n'
                f'学校教員に直接申請内容を伝えてください。'
            )
        return response

    if has_required and confirmed:
        return request_official_absence(date, first_period_class, second_period_class, third_period_class, fourth_period_class, fifth_period_class, reason)
    else:
        if has_required:
            return confirm_template
        else:
            return request_information_template


application_items_tools = [application_items]

memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True)
chat_history = MessagesPlaceholder(variable_name='chat_history')

# モデルの初期化
llm = AzureChatOpenAI(  # Azure OpenAIのAPIを読み込み。
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_GPT35_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
    model_kwargs={"top_p": 0.1, "function_call": {"name": "application_items"}}
)

agent_kwargs = {
    "system_message": SystemMessagePromptTemplate.from_template(template=APPLICATION_ITEMS_SYSTEM_PROMPT),
    "extra_prompt_messages": [chat_history]
}
official_absence_agent = initialize_agent(
    application_items_tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=verbose,
    agent_kwargs=agent_kwargs,
    memory=memory
)


messages = []
messages.extend(official_absence_agent.agent.prompt.messages[:3])
messages.append(SystemMessagePromptTemplate.from_template(
    template=APPLICATION_ITEMS_SSUFFIX_PROMPT),)
messages.append(official_absence_agent.agent.prompt.messages[3])
official_absence_agent.agent.prompt.messages = messages



# message = "公欠届を申請したいです。"
# print(official_absence_agent.run(message))
while True:
    message = input(">> ")
    if message == "exit":
        break
    response = official_absence_agent.run(message)
    print(response)
