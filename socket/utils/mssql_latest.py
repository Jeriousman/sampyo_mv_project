import os
import json
import pyodbc
from PIL import Image


class DB:
    def __init__(self, host, dbname, username, password):
        self.driver = "ODBC Driver 17 for SQL Server"
        self.host = host
        self.dbname = dbname
        self.username = username
        self.password = password
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

    def insert(self, data, vin_date, model_item, model_type,image_path, quantity):
        connection_string = f'DRIVER={{{self.driver}}};SERVER={self.host};DATABASE={self.dbname};UID={self.username};PWD={self.password}'
        connection = pyodbc.connect(connection_string)

        #vplant_code, vseq_no, car_num, item_code, from_code, type_code = data.split('|')
        if len(data[1:-1].split('|')) == 7:
            vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url = data[1:-1].split('|')

        if len(data[1:-1].split('|')) == 8:
            vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = data[1:-1].split('|')
        # remove STX, ETX
        #vplant_code = vplant_code[1:]
        #type_code = type_code[:-1]

        # Load image
        origin_image = Image.open(image_path)
        
        # Resize image
        temp_image_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/socket/temp.jpg'

        resized_image = origin_image.resize((192, 108))
        resized_image.save(temp_image_path)


        # Resize LPR Image for Thumbnail Image
        temp_lpr_image_path = '/home/sdt/Workspace/onvif/python-onvif-zeep/socket/temp_lpr.jpg'
        os.system(f"wget {img_url}")
        LPR_IMG_FILE_NAME = img_url.split("/")[-1]
        try:
            origin_LPR_image = Image.open(LPR_IMG_FILE_NAME)
            resized_LPR_image = origin_LPR_image.resize((192, 108))
            resized_LPR_image.save(temp_lpr_image_path)
        except Exception as e:
            origin_LPR_image = None
            resized_LPR_image = None
            #resized_LPR_image.save(temp_lpr_image_path)
 #           logger.error("{e}")


        ##os.remove(f'/home/sdt/Workspace/onvif/python-onvif-zeep/socket/{LPR_IMG_FILE_NAME}')


        # DB Insert
        with open(temp_image_path, 'rb') as f:
            thumbnail = f.read()

        try:
            with open(temp_lpr_image_path, 'rb') as f:
                lpr_thumbnail = f.read()
        except Exception as e:
            lpr_thumbnail = None
#            logger.error("{e}")



        query = f"INSERT INTO CLS_LOG (VIN_DATE, PLANT_CODE, VSEQ_NO, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, MODEL_TYPE, IMAGE_PATH, THUMBNAIL, QUANTITY, COMPANY, LPR_THUMBNAIL, LPR_THUMBNAIL_PATH) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
        params = (
                vin_date, 
                vplant_code, 
                vseq_no, 
                car_num, 
                self.categories['item'][item_code], 
                self.categories['type'][type_code], 
                model_item,
                model_type,
                image_path, 
                pyodbc.Binary(thumbnail),
                quantity,
                self.categories['from'][from_code], 
                pyodbc.Binary(lpr_thumbnail),
                img_url
        )

        cursor = connection.cursor()
        cursor.execute(query, params)
        connection.commit()

        os.remove(temp_image_path)
        os.remove(temp_lpr_image_path)

        cursor.close()
        connection.close()

    def update(self, str_data, item):
        vplant_code, vseq_no, *_ = str_data.split('|')
        vplant_code = vplant_code[1:]

        connection_string = f'DRIVER={{{self.driver}}};SERVER={self.host};DATABASE={self.dbname};UID={self.username};PWD={self.password}'
        connection = pyodbc.connect(connection_string)

        cursor = connection.cursor()
        query = f"UPDATE E_INOUTLOG SET SDT_ITEM='{item}' WHERE VPLANT_CODE='{vplant_code}' AND VSEQ_NO={vseq_no}"

        cursor.execute(query)
        connection.commit()

        cursor.close()
        connection.close()


