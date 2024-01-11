import os
from dotenv import load_dotenv
load_dotenv(override=True)


from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import OpenAIEmbeddings

from .school_db import select_data
from ..template import default_value

class Document:
    page_content: str
    metadata: dict

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata

def get_contexts(table_name):
    sql_data = select_data(table_name)
    return sql_data




def add_vector():
    vector_store_address: str = os.environ["AZURE_SEARCH_ENDPOINT"]
    vector_store_password: str = os.environ["AZURE_SEARCH_KEY"]
    embeddings: OpenAIEmbeddings = default_value.default_embeddings_model

    index_name: str = "vector-class-data"
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    
    sql_data = get_contexts("class_data")
    docs = []
    for data in sql_data:
        # リスト型に変換して追加
        doc = Document(data["data"], {"id": str(data["id"]), "class_name": data["class_name"]})
        docs.append(doc)
    print(docs)
    vector_store.add_documents(documents=docs)
    

def search_vector(index_name, search_word):
    vector_store_address: str = os.environ["AZURE_SEARCH_ENDPOINT"]
    vector_store_password: str = os.environ["AZURE_SEARCH_KEY"]
    embeddings: OpenAIEmbeddings = default_value.default_embeddings_model

    index_name: str = index_name
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=vector_store_address,
        azure_search_key=vector_store_password,
        index_name=index_name,
        embedding_function=embeddings.embed_query,
    )
    res = vector_store.similarity_search(query=search_word, search_type="hybrid")
    search_result = []
    for doc in res:
        if hasattr(doc, 'page_content'):
            search_result.append({
                'id': doc.metadata['id'],
                'class_name': doc.metadata['class_name'],
                'data': doc.page_content
            })
    return search_result


