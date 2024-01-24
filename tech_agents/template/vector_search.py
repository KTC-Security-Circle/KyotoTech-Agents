from typing import List
from langchain_community.vectorstores.azuresearch import AzureSearch

from tech_agents.template.models import Document
from tech_agents.template import default_value


def dilect_vector(index_name, documents):  # ベクトルdata追加関数
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=default_value.vector_store_address,
        azure_search_key=default_value.vector_store_password,
        index_name=index_name,
        embedding_function=default_value.embeddings.embed_query,
    )

    vector_store.add_documents(documents=documents)


def search_vector(
    index_name: str,
    search_word: str,
    k: int = 3
) -> List[Document]:  # ベクトル検索関数
    index_name: str = index_name
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=default_value.vector_store_address,
        azure_search_key=default_value.vector_store_password,
        index_name=index_name,
        embedding_function=default_value.embeddings.embed_query,
    )
    res = vector_store.similarity_search(
        query=search_word, search_type="hybrid", k=k)
    return res
