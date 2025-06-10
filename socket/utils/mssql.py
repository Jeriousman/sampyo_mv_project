import os
import json
import pyodbc
from PIL import Image

import logging
import logging.handlers
import traceback

######################################################################
#                             Save Log                               #
######################################################################
logger_db = logging.getLogger("logger_db")
logger_db.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log_max_size = 1024000
log_file_count = 3
log_fileHandler = logging.handlers.RotatingFileHandler(
        filename=f"/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/logs/socket_db_data.log",
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode='a')

log_fileHandler.setFormatter(formatter)
# logger_db.addHandler(log_fileHandler)

logger_db.addHandler(log_fileHandler)

# 부모 로거로 로그가 전파되지 않도록 설정 (루트 로거 영향 제거)
logger_db.propagate = False

# 로그 테스트
# logger_db.info("sql.py: INFO 로그 테스트")
# logger_db.error("sql.py: ERROR 로그 테스트")

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
        try:
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
            logger_db.info(f'img_url: {img_url}')
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
                logger_db.error(traceback.format_exc())

            ##os.remove(f'/home/sdt/Workspace/onvif/python-onvif-zeep/socket/{LPR_IMG_FILE_NAME}')

            # DB Insert
            with open(temp_image_path, 'rb') as f:
                thumbnail = f.read()

            try:
                with open(temp_lpr_image_path, 'rb') as f:
                    lpr_thumbnail = f.read()
            except Exception as e:
                lpr_thumbnail = None
                logger_db.error(traceback.format_exc())

            query = f"INSERT INTO CLS_LOG (VIN_DATE, PLANT_CODE, VSEQ_NO, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, MODEL_TYPE, IMAGE_PATH, THUMBNAIL, COMPANY, LPR_THUMBNAIL, LPR_THUMBNAIL_PATH ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"
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
                    self.categories['from'][from_code], 
                    pyodbc.Binary(lpr_thumbnail),
                    img_url
            )

            cursor = connection.cursor()
            cursor.execute(query, params)
            connection.commit()
            logger_db.info('DB Insert Committ!')

            # cursor.execute(f"""
            # UPDATE CLS_LOG
            # SET VIN_DATE = {vin_date},
            # SET PLANT_CODE = {vplant_code},
            # SET CAR_NUM = {car_num},
            # SET ITEM_NAME = {self.categories['item'][item_code]},
            # SET ITEM_TYPE = {self.categories['type'][type_code]},
            # SET MODEL_ITEM = {model_item},
            # SET MODEL_TYPE = {model_type},
            # SET IMAGE_PATH = {image_path},
            # SET THUMBNAIL = {pyodbc.Binary(thumbnail)},
            # SET COMPANY = {self.categories['from'][from_code]},
            # SET LPR_THUMBNAIL = {pyodbc.Binary(lpr_thumbnail)},
            # SET LPR_THUMBNAIL_PATH = {img_url},

            # WHERE VSEQ_NO = {vseq_no};

            # INSERT INTO CLS_LOG (VIN_DATE, PLANT_CODE, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, MODEL_TYPE, IMAGE_PATH, THUMBNAIL, COMPANY, LPR_THUMBNAIL, LPR_THUMBNAIL_PATH)
            # SELECT {vin_date}, {vplant_code}, {car_num}, {self.categories['item'][item_code]}, {self.categories['type'][type_code]}, {model_item}, {model_type}, {image_path}, {pyodbc.Binary(thumbnail)}, {self.categories['from'][from_code]}, {pyodbc.Binary(lpr_thumbnail)}, {img_url}
            # WHERE NOT EXISTS (
            #     SELECT 1
            #     FROM CLS_LOG
            #     WHERE VSEQ_NO = {vseq_no}
            # );
            # """)
            os.remove(f'/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/{LPR_IMG_FILE_NAME}')
            #os.remove(img_url)
            os.remove(temp_image_path)
            os.remove(temp_lpr_image_path)

            cursor.close()
            connection.close()
        except:
            logger_db.error(traceback.format_exc())

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

    
    def upsert_ptz(self, data, vin_date, model_item, model_type,image_path):
        logger_db.info('in upsert ptz!')
        try:
            connection_string = f'DRIVER={{{self.driver}}};SERVER={self.host};DATABASE={self.dbname};UID={self.username};PWD={self.password}'
            connection = pyodbc.connect(connection_string)
            logger_db.info(f"Conectione to DB: {connection}")
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
                logger_db.error(traceback.format_exc())
        
            ##os.remove(f'/home/sdt/Workspace/onvif/python-onvif-zeep/socket/{LPR_IMG_FILE_NAME}')

            # DB Insert
            with open(temp_image_path, 'rb') as f:
                thumbnail = f.read()

            try:
                with open(temp_lpr_image_path, 'rb') as f:
                    lpr_thumbnail = f.read()
            except Exception as e:
                lpr_thumbnail = None
                logger_db.error(traceback.format_exc())


            ####################[1]
            # insert_sql = f"""
            # INSERT IGNORE INTO CLS_LOG (VSEQ_NO, VIN_DATE, PLANT_CODE, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, MODEL_TYPE, IMAGE_PATH, THUMBNAIL, COMPANY, LPR_THUMBNAIL, LPR_THUMBNAIL_PATH) 
            # VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            # """

            # update_sql = f"""
            # UPDATE CLS_LOG 
            # SET VIN_DATE = ? 
            # SET PLANT_CODE = ? 
            # SET CAR_NUM = ? 
            # SET ITEM_NAME = ? 
            # SET ITEM_TYPE = ? 
            # SET MODEL_ITEM = ? 
            # SET MODEL_TYPE = ? 
            # SET IMAGE_PATH = ? 
            # SET THUMBNAIL = ? 
            # SET COMPANY = ? 
            # SET LPR_THUMBNAIL = ? 
            # SET LPR_THUMBNAIL_PATH = ? 

            # WHERE VSEQ_NO = ?;
            # """

            # # Insert if not exists
            # cursor.execute(f"{insert_sql}, {(vseq_no, vin_date, vplant_code, 
            #                                  car_num, self.categories['item'][item_code],
            #                                  self.categories['type'][type_code], model_item,
            #                                  model_type, image_path, pyodbc.Binary(thumbnail),
            #                                  self.categories['from'][from_code], pyodbc.Binary(lpr_thumbnail),
            #                                  img_url 
            #                                  )}")

            # # Update after insert
            # cursor.execute(f"{update_sql}, {(vin_date, vplant_code, car_num,
            #                                  self.categories['item'][item_code], self.categories['type'][type_code],
            #                                  model_item, model_type, image_path, pyodbc.Binary(thumbnail),
            #                                  self.categories['from'][from_code], pyodbc.Binary(lpr_thumbnail),
            #                                  img_url, vseq_no)}")

            cursor = connection.cursor()
            
            #############[2]
            # Step 1: Try updating the existing row
            update_sql = f"""
            UPDATE CLS_LOG
            SET VIN_DATE = ?, PLANT_CODE = ?, CAR_NUM = ?, ITEM_NAME = ?, ITEM_TYPE = ?, MODEL_ITEM = ?, MODEL_TYPE = ?, IMAGE_PATH = ?, THUMBNAIL = ?, COMPANY = ?, LPR_THUMBNAIL = ?, LPR_THUMBNAIL_PATH = ?
            WHERE VSEQ_NO = ?;
            """
            cursor.execute(update_sql, (vin_date, vplant_code, car_num, self.categories['item'][item_code], self.categories['type'][type_code], model_item, model_type, image_path, pyodbc.Binary(thumbnail), self.categories['from'][from_code], pyodbc.Binary(lpr_thumbnail), img_url, vseq_no))
            connection.commit()
            #logger_db.info(f"DB row update: {cursor.execute(upsert_sql, (vin_date, vplant_code, car_num, self.categories['item'][item_code], self.categories['type'][type_code], model_item, model_type, image_path, pyodbc.Binary(thumbnail), self.categories['from'][from_code], pyodbc.Binary(lpr_thumbnail), img_url, vseq_no))}")

            # # Step 2: If no rows were updated, insert a new row
            if cursor.rowcount == 0 or cursor.rowcount == -1:
                insert_sql = f"""
                INSERT INTO CLS_LOG (VSEQ_NO, VIN_DATE, PLANT_CODE, CAR_NUM, ITEM_NAME, ITEM_TYPE, MODEL_ITEM, MODEL_TYPE, IMAGE_PATH, THUMBNAIL, COMPANY, LPR_THUMBNAIL, LPR_THUMBNAIL_PATH) 
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ? ,? ,? ,?);
                """
                cursor.execute(insert_sql, (vseq_no, vin_date, vplant_code, car_num, self.categories['item'][item_code], self.categories['type'][type_code], model_item, model_type, image_path,  pyodbc.Binary(thumbnail), self.categories['from'][from_code], pyodbc.Binary(lpr_thumbnail), img_url))
                #logger_db.info(f"DB row insert: {cursor.execute(insert_sql, (vseq_no, vin_date, vplant_code, car_num, self.categories['item'][item_code], self.categories['type'][type_code], model_item, model_type, image_path,  pyodbc.Binary(thumbnail), self.categories['from'][from_code], pyodbc.Binary(lpr_thumbnail), img_url))}")
                connection.commit()
            # cursor.execute(query, params)
            
            logger_db.info('upsert_ptz commit!')

            os.remove(f'/home/sdt/Workspace/onvif/python-onvif-zeep/socket/sam2/{LPR_IMG_FILE_NAME}')
            os.remove(temp_image_path)
            os.remove(temp_lpr_image_path)

            cursor.close()
            connection.close()
        except:
                logger_db.error(traceback.format_exc())


    def upsert_depth(self, data, quantity):
        logger_db.info('=====in upsert_depth========')

        try:
            connection_string = f'DRIVER={{{self.driver}}};SERVER={self.host};DATABASE={self.dbname};UID={self.username};PWD={self.password}'
            connection = pyodbc.connect(connection_string)

            #vplant_code, vseq_no, car_num, item_code, from_code, type_code = data.split('|')
            if len(data[1:-1].split('|')) == 7:
                vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url = data[1:-1].split('|')

            if len(data[1:-1].split('|')) == 8:
                vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = data[1:-1].split('|')


            cursor = connection.cursor()





            #############[0]
            # cursor.execute(f"""
            # UPDATE CLS_LOG
            # SET QUANTITY = {quantity},
            # # SET VSEQ_NO = {vseq_no},
            # WHERE VSEQ_NO = {vseq_no};

            # INSERT INTO CLS_LOG (QUANTITY)
            # SELECT {quantity}
            # WHERE NOT EXISTS (
            #     SELECT 1
            #     FROM CLS_LOG
            #     WHERE VSEQ_NO = {vseq_no}
            # );
            # """)




            #############[1]
            # insert_sql = f"""
            # INSERT IGNORE INTO CLS_LOG (VSEQ_NO, QUANTITY) 
            # VALUES (?, ?);
            # """
            # update_sql = f"""
            # UPDATE CLS_LOG 
            # SET QUANTITY = ? 
            # WHERE VSEQ_NO = ?;
            # """

            # # Insert if not exists
            # cursor.execute(f"{insert_sql}, {(vseq_no, quantity)}")

            # # Update after insert
            # cursor.execute(f"{update_sql}, {(quantity, vseq_no)}")





            ##############[2]
            # Step 1: Try updating the existing row
            update_sql = f"""
            UPDATE CLS_LOG
            SET QUANTITY = ? 
            WHERE VSEQ_NO = ?;
            """
            cursor.execute(update_sql, (quantity, vseq_no))

            # Step 2: If no rows were updated, insert a new row
            if cursor.rowcount == 0 or cursor.rowcount == -1:
                insert_sql = f"""
                INSERT INTO CLS_LOG (VSEQ_NO, QUANTITY) 
                VALUES (?, ?);
                """
                cursor.execute(insert_sql, (vseq_no, quantity))


            
            connection.commit()
            logger_db.info('upsert_depth commit!')


            cursor.close()
            connection.close()
        except:
            logger_db.error(traceback.format_exc())





    def upsert_mv(self, data, granular_info):

        logger_db.info('=====in upsert_mv========')
        try:
            connection_string = f'DRIVER={{{self.driver}}};SERVER={self.host};DATABASE={self.dbname};UID={self.username};PWD={self.password}'
            connection = pyodbc.connect(connection_string)

            #vplant_code, vseq_no, car_num, item_code, from_code, type_code = data.split('|')
            if len(data[1:-1].split('|')) == 7:
                vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url = data[1:-1].split('|')

            if len(data[1:-1].split('|')) == 8:
                vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = data[1:-1].split('|')

            cursor = connection.cursor()

            # #############[0]
            # cursor.execute(f"""
            # UPDATE CLS_LOG
            # SET SIZE_ANLZ = {granular_info},
            # WHERE VSEQ_NO = {vseq_no};

            # INSERT INTO CLS_LOG (SIZE_ANLZ)
            # SELECT {granular_info}
            # WHERE NOT EXISTS (
            #     SELECT 1
            #     FROM CLS_LOG
            #     WHERE VSEQ_NO = {vseq_no}
            # );
            # """)

            ###########[1]
            # insert_sql = f"""
            # INSERT IGNORE INTO CLS_LOG (VSEQ_NO, SIZE_ANLZ) 
            # VALUES (?, ?);
            # """
            # update_sql = f"""
            # UPDATE CLS_LOG 
            # SET SIZE_ANLZ = ? 
            # WHERE VSEQ_NO = ?;
            # """

            # # Insert if not exists
            # cursor.execute(f"{insert_sql}, {(vseq_no, granular_info)}")

            # # Update after insert
            # cursor.execute(f"{update_sql}, {(granular_info, vseq_no)}")

            # ########[2]
            # # Step 1: Try updating the existing row
            # update_sql = f"""
            # UPDATE CLS_LOG 
            # SET SIZE_ANLZ = ? 
            # WHERE VSEQ_NO = ?;
            # """
            # cursor.execute(update_sql, (granular_info, vseq_no))

            # # Step 2: If no rows were updated, insert a new row
            # if cursor.rowcount == 0 or cursor.rowcount == -1:
            #     insert_sql = f"""
            #     INSERT INTO CLS_LOG (VSEQ_NO, SIZE_ANLZ) 
            #     VALUES (?, ?);
            #     """
            #     cursor.execute(insert_sql, (vseq_no, granular_info))

            ##########[3]
            upsert_sql = """
            IF EXISTS (SELECT 1 FROM CLS_LOG WHERE VSEQ_NO = ?)
                UPDATE CLS_LOG 
                SET SIZE_ANLZ = ?
                WHERE VSEQ_NO = ?
            ELSE
                INSERT INTO CLS_LOG (VSEQ_NO, SIZE_ANLZ)
                VALUES (?, ?);
            """

            vplant_code, vseq_no, car_num, item_code, from_code, type_code, img_url, _ = data[1:-1].split('|')
            cursor.execute(upsert_sql, (vseq_no, str(granular_info), vseq_no,
                                        vseq_no, str(granular_info)))

            connection.commit()
            logger_db.info('upsert_mv commit!')



            # connection.commit()

            cursor.close()
            connection.close()
        except:
            logger_db.error(traceback.format_exc())

