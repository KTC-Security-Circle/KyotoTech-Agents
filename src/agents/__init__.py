import main
from . import tools

if __name__ == "__main__":
    # ユーザーの入力を受け取る
    user_message = input("User >> ")
    # ユーザーの入力を引数にして実行する
    result = main.run(user_message)
    # 結果を表示する
    print("システム応答 : ", result)