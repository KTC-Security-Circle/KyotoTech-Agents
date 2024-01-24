from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter

from tech_agents.db import vector, school_db, models


# loader = DirectoryLoader('シラバス/text/', glob="**/*.txt")
# docs = loader.load()

# text_splitter = CharacterTextSplitter(
#     separator=" ",
#     chunk_size=50,
#     chunk_overlap=0,
#     length_function=len,
#     is_separator_regex=False,
# )
# i = 0
# meta = [
#     "python機械学習", "WEBフロントエンド実践開発", "データ分析応用"
# ]
# for doc in docs:
#     split_docs = text_splitter.create_documents([doc.page_content])
#     metadata = doc.metadata
#     metadata["class_name"] = meta[i]
#     print(metadata)
#     print(split_docs)
#     documents = []


# for doc in docs:
#     split_docs = text_splitter.create_documents([doc.page_content])
#     metadata = doc.metadata
#     documents = []
#     for split_doc in split_docs:
#         split_doc.metadata = metadata
#         split_doc.metadata["split_source"] = split_doc.page_content
#         documents.append(doc)

