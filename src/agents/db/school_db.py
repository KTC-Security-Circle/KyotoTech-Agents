import os
from dotenv import load_dotenv
import pyodbc


load_dotenv(override=True)

server = os.environ.get('db_host')
database = os.environ.get('db_name')
username = os.environ.get('db_user')
password = os.environ.get('db_pas')
driver = '{ODBC Driver 18 for SQL Server}'


def get_contexts(user_id):
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f"SELECT TOP (3) id, CONVERT(datetime, created_at) as created_at, question, answer, user_id FROM [dbo].[chat_app_chatlog] WHERE user_id =  {user_id}  ORDER BY created_at DESC")
            row = cursor.fetchone()
            contexts = []
            while row:
                # 存在する属性にアクセスする前にチェック
                if hasattr(row, 'question') and hasattr(row, 'answer'):
                    contexts.insert(0, {
                        'question': row.question,
                        'answer': row.answer
                    })
                    # print("Question: {}, Answer: {}".format(row.question, row.answer))
                row = cursor.fetchone()
    return contexts


def create_table():  # テーブル作成関数
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                '''
                    CREATE TABLE class_data
                    (
                        id int IDENTITY(1,1) PRIMARY KEY,
                        class_name NVARCHAR(20) NOT NULL,
                        data NVARCHAR(MAX) NOT NULL,
                    )
                ''')
            cursor.commit()


def insert_data(class_name, data):  # データ挿入関数
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                '''
                    INSERT INTO class_data (class_name, data)
                    VALUES (?, ?)
                ''', class_name, data)
            cursor.commit()


def select_data(table_name):  # データ取得関数
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                f'SELECT TOP (1000) * FROM [dbo].[{table_name}]')
            row = cursor.fetchone()
            contexts = []
            while row:
                print(row)
                # 存在する属性にアクセスする前にチェック
                if hasattr(row, 'id') and hasattr(row, 'class_name') and hasattr(row, 'data'):
                    contexts.append({
                        'id': row.id,
                        'class_name': row.class_name,
                        'data': row.data
                    })
                    # print("Question: {}, Answer: {}".format(row.question, row.answer))
                row = cursor.fetchone()
            return contexts


def drop_table():
    with pyodbc.connect('DRIVER='+driver+';SERVER=tcp:'+server+';PORT=1433;DATABASE='+database+';UID='+username+';PWD=' + password) as conn:
        with conn.cursor() as cursor:
            cursor.execute('DROP TABLE class_data')
            cursor.commit()


# create_table()
# drop_table()

# insert_data(class_name='Python機械学習', data='Python機械学習のテストデータです')

# data = select_data('class_data')
# print(data)
