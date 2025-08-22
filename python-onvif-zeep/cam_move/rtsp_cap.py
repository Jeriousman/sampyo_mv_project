import cv2
from datetime import datetime

# RTSP 링크 설정
rtsp_link = "rtsp://admin:sampyo123!@172.17.16.6:558/LiveChannel/0/media.smp"

# OpenCV VideoCapture 객체 생성
cap = cv2.VideoCapture(rtsp_link)
cap.set(3, 1920)  # 영상 가로길이 설정
cap.set(4, 1080)  # 영상 세로길이 설정
fps = 20 


# RTSP 스트림이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("Error: Failed to open RTSP stream")
    exit()

# 이미지를 캡처하는 함수
def capture_image():
    ret, frame = cap.read()  # 프레임 읽기
    if ret:
        return frame  # 캡처된 프레임 반환
    else:
        print("Error: Failed to capture frame")
        return None

# 캡처할 이미지를 저장할 경로
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
save_path = f"captured_image_{current_time}.jpg"

# 이미지 캡처 및 저장
image = capture_image()
if image is not None:
    image = cv2.resize(image, (1920, 1080))
    cv2.imwrite(save_path, image)
    print(f"Image captured and saved to {save_path}")
else:
    print("Failed to capture image")

# VideoCapture 객체 해제
cap.release()

