import os, sys

from langchain.memory import ConversationBufferMemory

import tech_agents

# memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
# memory.save_context({"input": "こんにちは、私の名前は遠藤です。"}, {
#                     "output": "こんにちは遠藤さん！なんのご用ですか？"})

# デバッグ先の指定
# agent = agents.MainAgent(memory=memory, verbose=True)
agent = tech_agents.MainAgent(verbose=True)
# agent = tools.ProcedureAgent(verbose=True)

# debug code
##### debug前に pip install . を実行 #####
while True:
    message = input(">> ")
    if message == "exit" or message == "終了":
        print("会話履歴を保存しますか。（Y/n）")
        input_message = input(">> ")
        if input_message == "Y" or input_message == "y" or input_message == "yes" or input_message == "Yes":
            try:
                buffer = agent.memory.load_memory_variables({})
                with open("chat_history.txt", mode="w", encoding='utf-8') as f:
                    f.write(f'{buffer["chat_history"]}')
                    print("会話履歴を保存しました。")
            except Exception as e:
                print("err : " + str(e))
        print("終了します。")
        break

    ai_response = agent.run(message)
    print(f'AI : {ai_response}')


