# 🤖 マルチターン・マルチタスクAI - Powered by Azure OpenAI

このパッケージは、`Azure OpenAI` と `LangChain` を活用して、ファインチューニングせずに特定の分野や企業専用のマルチターン・マルチタスクAIを作成するためのものです。このプロジェクトでは、複数のタスクを効率的に処理し、柔軟な対話型AIアプリケーションを実現するためのツールとフレームワークを提供します。

## 📚 目次

- [🤖 マルチターン・マルチタスクAI - Powered by Azure OpenAI](#-マルチターンマルチタスクai---powered-by-azure-openai)
  - [📚 目次](#-目次)
  - [⭐ 特徴](#-特徴)
  - [💻 インストール](#-インストール)
  - [🌍 環境変数](#-環境変数)
  - [🚀 使い方](#-使い方)
  - [🛠️ テクノロジ](#️-テクノロジ)

## ⭐ 特徴

- **Function Callingの統合**: Azure OpenAIの高度なFunction Calling機能を利用して、複数のタスクを必要かどうかをAIが考え、適切に処理します。
- **モジュラー設計**: 異なるタイプのタスクを処理するためのモジュールが用意されており、新たにタスクを増やすことやカスタマイズが容易です。
- **効率的なデータ処理**: `langchain`ライブラリを活用し、データ処理の効率を最大化します。

## 💻 インストール

pip を使用して、このパッケージをインストールします。

```bash

pip install git+https://github.com/KTC-Security-Circle/KyotoTech-Agents.git
```

このパッケージはpipでは公開しておらず、Githubから直接インストールします。

このパッケージに必要なその他パッケージのバージョンに関してはrequirements.txtを参照してください。

## 🌍 環境変数

このパッケージでは、いくつかの環境変数を使用しています。これらの環境変数を設定する必要があります。
使用している環境変数については、[環境変数のリストと説明](docs/envList.md)を参照してください。

## 🚀 使い方

使用する前に、[環境変数のリストと説明](docs/envList.md)を参照して、環境変数を設定してください。

とりあえず動かしてみる場合：

```python
import agents

# エージェントの初期化
agent = agents.MainAgent()
print(agent.run("こんにちは"))
```

エージェントをカスタマイズしたい場合：

```python
import agents

# カスタムエージェントの初期化
custom_agent = agents.MainAgent(
  llm=custom_llm, # langchainのカスタムLLMを指定
  memory=custom_memory, # カスタムメモリを指定
  chat_history=custom_chat_history, # カスタムチャット履歴を指定
  verbose=True # デバッグモードを有効にする
)
print(custom_agent.run("こんにちは"))
```

この使い方の例では、`agents`モジュールの`MainAgent`クラスを使用しています。このクラスは、`langchain`ライブラリを使用して、ユーザーの入力を処理し、適切なタスクを選択して実行します。このクラスは、`llm`、`memory`、`chat_history`、`verbose`の4つのパラメータを受け取ります。これらのパラメータは、それぞれ、`langchain`のカスタムLLM、カスタムメモリ、カスタムチャット履歴、デバッグモードを有効にするかどうかを指定します。これらのパラメータは、すべてオプションです。これらのパラメータを指定しない場合、デフォルトの値が使用されます。
デフォルトの値は、`langchain`の`AzureChatOpenAI`、`langchain`の`ConversationBufferMemory`、`langchain`の`MessagesPlaceholder`、デバッグモードが無効になっていることです。

## 🛠️ テクノロジ

- Python
- langchain
- Azure OpenAI
- その他のPython標準ライブラリ
