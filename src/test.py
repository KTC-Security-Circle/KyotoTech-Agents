from agents.db import vector, school_db
from agents.db.school_db import insert_data
from langchain_community.document_loaders import PyPDFLoader


# SQLDBの操作関数
# テーブル作成 school_db.create_table(作成するテーブル名)
# school_db.create_table("class_data")
# school_db.create_table("scholarship_data")

# データ挿入 school_db.insert_data(テーブル名, データ, メタデータ)
## class_dataの例
# metadata = {
#     'class_name': 'python'
# }
# school_db.insert_data(table_name='class_data', data='pythonのテストデータです', metadata=metadata)

## scholarship_dataの例
# loader = PyPDFLoader("2023_taiyo_syougakukin_shiori.pdf")
# pages = loader.load_and_split()
# docs = pages[0:5]
# for doc in docs:
#     insert_data(table_name='scholarship_data',
#                 data=doc.page_content, metadata=doc.metadata)


# テーブル削除
# school_db.drop_table("class_data")
# school_db.drop_table("scholarship_data")



# ベクターストアの操作関数 
# ベクターストア作成 vector.add_vector(作成したいSQLDBのテーブル名)
# vector.add_vector("class_data")
# vector.add_vector("scholarship_data")

# ベクターストア検索
# res = vector.search_vector("vector-scholarship-data", "奨学金について")
# for doc in res:
#     print(doc.page_content, doc.metadata)
