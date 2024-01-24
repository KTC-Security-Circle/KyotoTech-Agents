from tech_agents.db import vector, school_db, models
from langchain.text_splitter import CharacterTextSplitter
from tech_agents.db.school_db import insert_data
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document


# SQLDBの操作関数
# テーブル作成 school_db.create_table(作成するテーブル名)
# school_db.create_table("class_data")
# school_db.create_table("scholarship_data")

# データ挿入 school_db.insert_data(テーブル名, データ, メタデータ)
# class_dataの例
# metadata = {
#     'class_name': 'python'
# }
# school_db.insert_data(table_name='class_data', data='pythonのテストデータです', metadata=metadata)

# scholarship_dataの例
# loader = PyPDFLoader("2023_taiyo_syougakukin_shiori.pdf")
# pages = loader.load_and_split()
# docs = pages[0:5]
# for doc in docs:
#     insert_data(table_name='scholarship_data',
#                 data=doc.page_content, metadata=doc.metadata)


# python_dataの例
# with open("シラバス/text/2023後期シラバス_月12_Python機械学習_木元先生.txt") as f:
#     file_data1 = f.read()

# with open("シラバス/text/2023後期シラバス_水3_データ分析応用_伊藤先生.txt") as f:
#     file_data2 = f.read()

# with open("シラバス/text/2023後期シラバス_月3_WEBフロントエンド実戦開発_佐野先生.txt") as f:
#     file_data3 = f.read()

# metadatas = [
#     {'source': '2023後期シラバス_月12_Python機械学習_木元先生.txt', 'class_name': 'Python機械学習'},
#     {'source': '2023後期シラバス_水3_データ分析応用_伊藤先生.txt', 'class_name': 'データ分析応用'},
#     {'source': '2023後期シラバス_月3_WEBフロントエンド実戦開発_佐野先生.txt', 'class_name': 'WEBフロントエンド実践開発'}
# ]

# text_splitter = CharacterTextSplitter(
#     separator="\n",
#     chunk_size=50,
#     chunk_overlap=0,
#     length_function=len,
#     is_separator_regex=False,
# )

# split_docs = text_splitter.create_documents(
#     [file_data1, file_data2, file_data3], metadatas=metadatas)


# for doc in split_docs:
#     doc.metadata["split_source"] = doc.page_content

# vector.dilect_vector("vector-class-data", split_docs)


# テーブル削除
# school_db.drop_table("class_data")
# school_db.drop_table("scholarship_data")


# ベクターストアの操作関数
# ベクターストア作成 vector.add_vector(作成したいSQLDBのテーブル名)
# vector.add_vector("class_data")
# vector.add_vector("scholarship_data")

# ベクターストア検索
# res = vector.search_vector("vector-class-data", "python", k=10)
# for doc in res:
#     print(doc)
