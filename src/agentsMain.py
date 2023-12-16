import os

from langchain.retrievers import AzureCognitiveSearchRetriever

retriever = AzureCognitiveSearchRetriever(
    service_name=os.environ["AZURE_COGNITIVE_SEARCH_SERVICE_NAME"],
    index_name=os.environ["AZURE_COGNITIVE_SEARCH_INDEX_NAME"],
    api_key=os.environ["AZURE_SEARCH_KEY"],
    content_key="content",
    top_k=2
)

res = retriever.get_relevant_documents("専攻について")

with open('output.txt', 'w') as f:
    for doc in res:
        # 各ドキュメントの page_content の抽出と書き込み
        if hasattr(doc, 'page_content'):
            f.write(doc.page_content + '\n')
