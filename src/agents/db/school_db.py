import os
from dotenv import load_dotenv
from typing import List
import pyodbc

from .models import Document


load_dotenv(override=True)

server = os.environ.get('db_host')
database = os.environ.get('db_name')
username = os.environ.get('db_user')
password = os.environ.get('db_pas')
driver = '{ODBC Driver 18 for SQL Server}'




def create_table(table_name):  # テーブル作成関数
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            query = f'''
                CREATE TABLE {table_name}
                (
                    id int IDENTITY(1,1) PRIMARY KEY,
                    data NVARCHAR(MAX) NOT NULL,
                    metadata NVARCHAR(MAX) NULL
                )
            '''

            cursor.execute(query)
            cursor.commit()



def insert_data(table_name, data, metadata):  # データ挿入関数
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            query = f'''
                INSERT INTO {table_name} (data, metadata)
                VALUES (?, ?)
            '''
            cursor.execute(query, data, str(metadata))
            cursor.commit()


def select_data(table_name) -> List[Document]:  # データ取得関数
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f'SELECT TOP (1000) * FROM [dbo].[{table_name}]')
            row = cursor.fetchone()
            contexts = []
            while row:
                # 存在する属性にアクセスする前にチェック
                if hasattr(row, 'id') and hasattr(row, 'data') and hasattr(row, 'metadata'):
                    metadata = eval(row.metadata)
                    doc = Document(row.data, metadata)
                    contexts.append(doc)
                row = cursor.fetchone()
    return contexts


def drop_table(table_name):
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(f'DROP TABLE {table_name}')
            cursor.commit()


