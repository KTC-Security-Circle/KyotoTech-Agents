from openai import InvalidRequestError

from agents.tools.grobal import grobal_value as g
import agents.main as main


while True:
    message = input(">> ")
    if message == "exit" or message == "終了":
        print("会話履歴を保存しますか。（Y/n）")
        input_message = input(">> ")
        if input_message == "Y" or input_message == "y" or input_message == "yes" or input_message == "Yes":
            try:
                buffer = g.memory.load_memory_variables({})
                with open("chat_history.txt", mode="w") as f:
                    f.write(f'{buffer["chat_history"]}')
                    print("会話履歴を保存しました。")
            except Exception as e:
                print("err : " + str(e))
        print("終了します。")
        break
    try:
        ai_response = main.run(message)
        print(f'"AI : "{ai_response}')
    except InvalidRequestError as error:
        message = error.response.json()["error"]["message"]
        print("Hit error: ", message)
        break
    except Exception as e:
        print("err : " + str(e))
        break
