import pyodbc
import io
from PIL import Image

# SQL Server 연결 정보
server = 'localhost'
database = 'sampyo_classification'
username = 'sa'
password = 'sampyo123!'
driver = 'ODBC Driver 17 for SQL Server'

connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(connection_string)

# 이미지 파일을 읽어와 바이너리 데이터로 변환
image_path = '20240321-080958.jpg'

orig_img = Image.open(image_path)
resized_img = orig_img.resize((192, 108))

resized_img.save('./test.jpg')

with open('./test.jpg', 'rb') as f:
    image_data = f.read()


# 이미지를 삽입할 쿼리 작성
sql = "INSERT INTO CLS_LOG (VSEQ_NO, THUMBNAIL, COMPANY) VALUES (?, ?, ?)"
image_id = 1  # 이미지의 고유 식별자
params = ('12345', pyodbc.Binary(image_data), u"하")

# 쿼리 실행
cursor = conn.cursor()
cursor.execute(sql, params)
conn.commit()

# 연결 종료
conn.close()

print("이미지가 성공적으로 삽입되었습니다.")

