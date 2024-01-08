'''

agentsパッケージ

このパッケージには、いくつかのエージェントをまとめたマルチエージェントの実装が含まれています。

使用方法
- `MainAgent` クラスをインスタンス化し、`run` メソッドにユーザーからの入力を与えてください。
- インスタンス化時に、使用するエージェントの実装を指定することができます。
- 個々のエージェントの実装は `agents.tools` サブパッケージにあります。
- デフォルトのエージェントは、`agents.tools.default` で、`Agent` クラスを使用してインスタンス化し実行します。

'''
from agents.main import *
