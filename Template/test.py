# from ConversationalPromptTemplate import call_langchain
from promptTemplate import call_langchain

# chat_history = [('京都テックについて教えてください', '京都テックは学校法人滋慶コミュニケーションアートが運営する専門学校で、京都テザイン＆テクノロジー専門学校の略称です。京都テックには「スーパーAI＆テクノロジー科」と「デジタルクリエイター科」の2つの学科があり、18個の専攻があります。18個の専攻は4年制 と3年制があり、それぞれ学科として分かれています。また、京都テックの公式ホームページはhttps://www.kyoto-tech.ac.jpで、Eメールアドレスはinfo@kyoto-tech.ac.jp、電話番号は0120-109-525です。')]


result = call_langchain(query="専攻についておしえて")
print(result)
# chat_history = [(result["question"], result["answer"])]
# print(chat_history)

# result2 = call_langchain(query="8階の設備について教えてください。", history=chat_history)
# print(result2)
