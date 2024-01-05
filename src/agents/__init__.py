'''**agentsパッケージ**

このパッケージには、いくつかのエージェントをまとめたマルチエージェントの実装が含まれています。

# 使用方法
- 基本は `agents.run` 関数を使ってエージェントを実行します。
- その際、引数にはユーザーからの入力を与えてください。

- 個々のエージェントの実装は `agents.tools` サブパッケージにあります。

'''
from agents.main import MainAgent
from agents import tools





def run(input: str):
    return MainAgent.run(input)
