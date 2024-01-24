from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter




class Document:  # ドキュメントクラス
    page_content: str
    metadata: dict

    def __init__(self, page_content: str, metadata: dict):
        self.page_content = page_content
        self.metadata = metadata


class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、"]
        super().__init__(separators=separators, **kwargs)
        

