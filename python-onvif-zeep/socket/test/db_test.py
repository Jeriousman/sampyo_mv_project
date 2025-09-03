import time
import pyodbc

server = '172.17.16.14'
database = 'NEW_RC_IMAGE'
username = 'sdt'
password = 'sdt'

# server = 'localhost'
# database = 'sampyo_classification'
# username = 'sa'
# password = 'sampyo123!'

port = 1433
driver = 'ODBC Driver 17 for SQL Server'

start = time.time()
connection_string = f'DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={username};PWD={password}'

connection = pyodbc.connect(connection_string)

cursor = connection.cursor()

item = '모래'
# query = "SELECT SDT_ITEM, ITEM_NM FROM E_INOUTLOG WHERE VPLANT_CODE='D09' and VSEQ_NO=202403290015"
query = "SELECT * FROM E_INOUTLOG WHERE VPLANT_CODE='D09' and VSEQ_NO=202405020048"
# query = "SELECT * FROM E_INOUTLOG"
# query = f"UPDATE E_INOUTLOG SET SDT_ITEM='{item}' WHERE VPLANT_CODE='D09' AND VSEQ_NO=202405020006"

cursor.execute(query)
# connection.commit()

# find column name
# table_name = 'E_INOUTLOG'
# cursor.execute("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = ?", (table_name,))

end = time.time()
print(f'{(end-start)%60}s')

results = cursor.fetchall()
print(len(results))
for r in results:
    print(r)

# cursor.close()
# connection.close()
