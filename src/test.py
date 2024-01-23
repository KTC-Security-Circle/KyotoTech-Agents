from agents.db import vector, school_db, models
from langchain.text_splitter import CharacterTextSplitter
from agents.db.school_db import insert_data
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
# loader = TextLoader("2023後期シラバス_月12_Python機械学習_木元先生.txt")
# file_data = loader.load()
# print(file_data)
# with open("2023後期シラバス_月12_Python機械学習_木元先生.txt") as f:
#     file_data = f.read()
#     metadata = {'source': '2023後期シラバス_月12_Python機械学習_木元先生.txt', 'class_name': 'python機械学習'}
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=50,
#         chunk_overlap=0,
#         length_function=len,
#         is_separator_regex=False,
#     )

#     split_docs = text_splitter.create_documents([file_data])

#     documents = []
#     for doc in split_docs:
#         doc.metadata = metadata
#         documents.append(doc)


#     vector.dilect_vector("vector-class-data", documents)


# テーブル削除
# school_db.drop_table("class_data")
# school_db.drop_table("scholarship_data")


# ベクターストアの操作関数
# ベクターストア作成 vector.add_vector(作成したいSQLDBのテーブル名)
# vector.add_vector("class_data")
# vector.add_vector("scholarship_data")

# ベクターストア検索
res = vector.search_vector("vector-class-data", "python")
for doc in res:
    print(doc.page_content, doc.metadata["split_source"])
