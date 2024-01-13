import os
from dotenv import load_dotenv
load_dotenv(override=True)
from typing import Any, List
import logging

from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

from .school_db import select_data
from .models import Document, JapaneseCharacterTextSplitter
from ..template import default_value



def get_contexts(table_name): # データベースからの取得関数
    sql_data = select_data(table_name)
    return sql_data


def split_data(documents: List[Document]) -> List[Document]: # データ分割関数
    first_text_splitter = JapaneseCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )
    second_text_splitter = JapaneseCharacterTextSplitter(
        chunk_size=50,
        chunk_overlap=10,
    )

    first_text_split_result = first_text_splitter.split_documents(documents)
    
    for doc in first_text_split_result:
        doc.metadata['split_source'] = doc.page_content
        
    second_text_split_result = second_text_splitter.split_documents(first_text_split_result)
    
    return second_text_split_result



def add_vector(table_name): # ベクトルdata追加関数
    replace_name = table_name.replace("_", "-")
    index_name: str = "vector-" + replace_name
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=default_value.vector_store_address,
        azure_search_key=default_value.vector_store_password,
        index_name=index_name,
        embedding_function=default_value.embeddings.embed_query,
    )
    
    try:
        print("データベースからデータの取得を開始します。")
        sql_data = get_contexts(table_name)
        print("データベースからの取得に成功しました。")
    except Exception as e:
        print(e)
        return
    
    try:
        print("データの分割を開始します。")
        documents = split_data(sql_data)
        print("データの分割に成功しました。")
    except Exception as e:
        print(e)
        return

    try:
        print("ベクトルデータの追加を開始します。")
        vector_store.add_documents(documents=documents)
        print("ベクトルデータの追加に成功しました。")
    except Exception as e:
        print(e)
        return




def search_vector(index_name, search_word): # ベクトル検索関数
    index_name: str = index_name
    vector_store: AzureSearch = AzureSearch(
        azure_search_endpoint=default_value.vector_store_address,
        azure_search_key=default_value.vector_store_password,
        index_name=index_name,
        embedding_function=default_value.embeddings.embed_query,
    )
    res = vector_store.similarity_search(query=search_word, search_type="hybrid", k=3)
    return res


