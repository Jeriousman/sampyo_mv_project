import pyodbc
import io
from PIL import Image

# SQL Server 연결 정보
server = '10.110.31.120'
database = 'sampyo_classification'
username = 'sa'
password = 'sampyo123!'
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(connection_string)

query = 'SELECT THUMBNAIL FROM CLS_LOG WHERE VSEQ_NO=12345'

# 쿼리 실행
cursor = conn.cursor()
cursor.execute(query)

result = cursor.fetchone()

with open('./test.jpg', 'wb') as f:
    f.write(result.THUMBNAIL)


# 연결 종료
conn.close()

print("이미지가 성공적으로 삽입되었습니다.")
