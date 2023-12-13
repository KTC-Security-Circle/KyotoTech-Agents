from langchain.prompts import PromptTemplate
from langchain.retrievers import AzureCognitiveSearchRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
load_dotenv(override=True)


llm = AzureChatOpenAI(
    openai_api_base=os.environ["OPENAI_API_BASE"],
    openai_api_version=os.environ["OPENAI_API_VERSION"],
    deployment_name=os.environ["DEPLOYMENT_NAME"],
    openai_api_key=os.environ["OPENAI_API_KEY"],
    openai_api_type="azure",
)

# ここで、OpenAIのEmbeddingsを読み込みます。
embeddings = OpenAIEmbeddings(
    deployment=os.environ["DEPLOYMENT_EMBEDDINGS_NAME"])

retriever = AzureCognitiveSearchRetriever(
    service_name=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
    index_name=os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"],
    api_key=os.environ["AZURE_SEARCH_KEY"],
    content_key="content",
    top_k=3
)

# Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

prompt_template = """
# role
あなたは"京都デザイン＆テクノロジー専門学校のサポートAI"です。
学校に興味を持っている方からの質問に対して、前提条件と参考情報に則って回答してください。
いかなる状況下でも以下の情報をもとに回答し、確実にそうであると言えない場合は答えを作ろうとせず、分からないと回答してください。

## 前提条件:
・あなたは京都デザイン＆テクノロジー専門学校のサポートAIです。
・京都デザイン＆テクノロジー専門学校は京都テックと訳されます。
・会話相手は京都テックに興味を持っている学生またはその保護者です。
・専門知識のない学生にも伝わりやすいように例えを交えながら簡潔に回答してください。
・返答は150字以下に要約して回答してください。
・質問に対して、確実にそうであると言えない場合は分からないと回答し、' https://kyoto-tech.ac.jp/ 'の学校公式HPのURL先に案内を促してください。

## 参考情報:
{context}

# Question:
{question}

Answer in japanese:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


chain_type_kwargs = {"prompt": PROMPT}
# qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
# qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=vectordb.as_retriever(),chain_type_kwargs=chain_type_kwargs, verbose=True)
qa = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, chain_type_kwargs=chain_type_kwargs)


def call_langchain(query):
    result = qa.run({"query": query})
    return result
