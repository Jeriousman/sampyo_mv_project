import json
import pymysql


with open('/home/sdt/Workspace/onvif/python-onvif-zeep/socket/config/db_info.json', 'r') as f:
    info = json.load(f)


class DB:
    def __init__(self):
        self.host = info['my_host']
        self.dbname = info['my_dbname']
        self.username = info['my_username']
        self.password = info['my_password']
        self.categories = {"item": {"31013": "쇄석   13",
                                    "31019": "쇄석   19",
                                    "31025": "쇄석   25",
                                    "31067": "경량골재 #67",
                                    "32001": "모래"},
                           "type": {"S1": "석산",
                                    "B1": "발파석",
                                    "S2": "세척사(바다모래)"},
                           "from": {"0493870000": "현대기업(주)",
                                    "0669560000": "삼호개발(주)",
                                    "0712360000": "주식회사 경환산업",
                                    "0736040000": "발안산업개발(주)",
                                    "0791340000": "유진기업(주) 항만부두",
                                    "0892200000": "성진소재(주) 인천",
                                    "0935600000": "진현토건(주)",
                                    "1326700000": "주식회사 지음코리아",
                                    "1500680000": "(주)삼표산업 화성사업소",
                                    "1512120000": "(주)에스피네이처 예산사업소",
                                    "1512730000": "주식회사 에스피네이처 안성사업소",
                                    "1920700000": "서두산업(주)용인지점",
                                    "2298820000": "금우산업개발 주식회사",
                                    "RC10017": "발안산업개발(하룡)"}}

    def insert(self, data, vin_date, model_item, image_path):
        vplant_code, vseq_no, car_num, item_code, from_code, type_code = data.split('|')

        # remove head, tail
        vplant_code = vplant_code[1:]
        type_code = type_code[:-1]

        connection = pymysql.connect(host=self.host,
                                     user=self.username,
                                     password=self.password,
                                     database=self.dbname)

        cursor = connection.cursor()
        # insert_data = (vin_date, vplant_code, vseq_no, car_num, self.categories['item'][item_code], self.categories['type'][type_code], model_item, image_path)
        insert_data = (vin_date, vplant_code, vseq_no, car_num, self.categories['item'][item_code], self.categories['type'][type_code], model_item, image_path, self.categories['from'][from_code])
        
        # query = f"INSERT INTO CLS_LOG (VIN_DATE, PLANT_CODE, VSEQ_NO, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, IMAGE_PATH) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)"
        query = f"INSERT INTO CLS_LOG (VIN_DATE, PLANT_CODE, VSEQ_NO, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, IMAGE_PATH, COMPANY) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
 
        cursor.execute(query, insert_data)
        connection.commit()

        cursor.close()
        connection.close()
    
